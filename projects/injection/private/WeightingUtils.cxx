#include "SIREN/injection/WeightingUtils.h"
#include "SIREN/injection/PhaseSpaceChannel.h"

#include <set>                                                    // for set
#include <array>                                                  // for array
#include <vector>                                                 // for vector
#include <cmath>                                                  // for isfinite
#include <cstddef>                                                // for size_t
#include <string>                                                 // for to_string
#include <sstream>

#include "SIREN/utilities/Errors.h"                     // for WeightCalculationError
#include "SIREN/interactions/CrossSection.h"            // for Cro...
#include "SIREN/interactions/InteractionCollection.h"  // for Cro...
#include "SIREN/interactions/Decay.h"                   // for Decay
#include "SIREN/dataclasses/InteractionRecord.h"         // for Int...
#include "SIREN/dataclasses/InteractionSignature.h"      // for Int...
#include "SIREN/dataclasses/Particle.h"                  // for Par...
#include "SIREN/detector/DetectorModel.h"                   // for Ear...
#include "SIREN/detector/Coordinates.h"
#include "SIREN/geometry/Geometry.h"                     // for Geo...
#include "SIREN/math/Vector3D.h"                         // for Vec...
#include "SIREN/utilities/Constants.h"                   // for cm

namespace siren {
namespace injection {

using detector::DetectorPosition;
using detector::DetectorDirection;

namespace {

double ConvertFinalStateDensity(
    double density,
    siren::dataclasses::PhaseSpaceTopology source_topology,
    siren::dataclasses::PhaseSpaceMeasure const & source_measure,
    PhaseSpaceConvention const & target,
    siren::dataclasses::InteractionRecord const & record)
{
    if (source_topology != target.topology) {
        throw siren::utilities::MeasureCompatibilityError(
            "Final-state topology does not match the requested weighting "
            "convention [siren-docs: errors#measure-compat]");
    }
    if (source_measure == target.measure) return density;
    return ConvertDensity(
        density, source_measure, target.measure, source_topology, record);
}

void MergeConvention(
    PhaseSpaceConvention & result,
    bool & found,
    siren::dataclasses::PhaseSpaceTopology topology,
    siren::dataclasses::PhaseSpaceMeasure const & measure)
{
    if (!found) {
        result.topology = topology;
        result.measure = measure;
        found = true;
        return;
    }
    if (result.topology == topology && result.measure == measure) return;

    if (result.topology == topology &&
        PhaseSpaceDensityConvertible(
            topology, result.measure, measure)) {
        result.measure = measure;
        return;
    }
    if (result.topology == topology &&
        PhaseSpaceDensityConvertible(
            topology, measure, result.measure)) {
        return;
    }

    std::ostringstream oss;
    oss << "Multiple interaction models match one signature but declare "
        << "different final-state conventions ("
        << siren::dataclasses::PhaseSpaceTopologyName(result.topology) << "/"
        << siren::dataclasses::PhaseSpaceMeasureName(result.measure) << " and "
        << siren::dataclasses::PhaseSpaceTopologyName(topology) << "/"
        << siren::dataclasses::PhaseSpaceMeasureName(measure)
        << "); there is no common pointwise density measure "
        << "[siren-docs: errors#measure-compat]";
    throw siren::utilities::MeasureCompatibilityError(oss.str());
}

PhaseSpaceConvention ResolveSelectedFinalStateConvention(
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record)
{
    PhaseSpaceConvention result;
    bool found = false;

    for (auto const & decay : interactions->GetDecays()) {
        for (auto const & signature :
             decay->GetPossibleSignaturesFromParent(
                 record.signature.primary_type)) {
            if (signature != record.signature) continue;
            MergeConvention(
                result, found, decay->TopologyForSignature(signature),
                decay->MeasureForSignature(signature));
        }
    }
    for (auto const & target_xs : interactions->GetCrossSectionsByTarget()) {
        for (auto const & cross_section : target_xs.second) {
            for (auto const & signature :
                 cross_section->GetPossibleSignaturesFromParents(
                     record.signature.primary_type, target_xs.first)) {
                if (signature != record.signature) continue;
                MergeConvention(
                    result, found,
                    cross_section->TopologyForSignature(signature),
                    cross_section->MeasureForSignature(signature));
            }
        }
    }
    return result;
}

// Shared rate-accumulation logic used by all probability functions.
// Returns (total_rate, selected_channel_rate).
// The selected_channel_rate is the sum of rates for all channels
// whose signature matches the record's signature.
// When candidate_count is non-null it receives the number of candidate
// signatures enumerated (the number of competing channels), so a caller can
// distinguish the single-channel case without a second traversal.
std::pair<double, double> AccumulateRates(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record,
    std::size_t * candidate_count = nullptr)
{
    std::size_t candidates = 0;
    siren::math::Vector3D interaction_vertex(
            record.interaction_vertex[0],
            record.interaction_vertex[1],
            record.interaction_vertex[2]);

    siren::math::Vector3D primary_direction(
            record.primary_momentum[1],
            record.primary_momentum[2],
            record.primary_momentum[3]);
    primary_direction.normalize();

    siren::geometry::Geometry::IntersectionList intersections =
        detector_model->GetIntersections(
            DetectorPosition(interaction_vertex),
            DetectorDirection(primary_direction));

    std::set<siren::dataclasses::ParticleType> const & possible_targets =
        interactions->TargetTypes();
    std::set<siren::dataclasses::ParticleType> available_targets =
        detector_model->GetAvailableTargets(
            intersections, DetectorPosition(record.interaction_vertex));

    double total_rate = 0.0;
    double selected_rate = 0.0;
    siren::dataclasses::InteractionRecord fake_record = record;

    // Decays
    for (auto const & decay : interactions->GetDecays()) {
        for (auto const & signature :
             decay->GetPossibleSignaturesFromParent(
                 record.signature.primary_type)) {
            ++candidates;
            fake_record.signature = signature;
            double decay_prob = 1.0 / (
                decay->TotalDecayLength(fake_record)
                / siren::utilities::Constants::cm);
            total_rate += decay_prob;
            if (signature == record.signature) {
                selected_rate += decay_prob;
            }
        }
    }

    // Cross sections
    for (auto const target : available_targets) {
        if (possible_targets.find(target) == possible_targets.end())
            continue;
        double target_density =
            detector_model->GetParticleDensity(
                intersections, DetectorPosition(interaction_vertex), target);
        for (auto const & cross_section :
             interactions->GetCrossSectionsForTarget(target)) {
            for (auto const & signature :
                 cross_section->GetPossibleSignaturesFromParents(
                     record.signature.primary_type, target)) {
                ++candidates;
                fake_record.signature = signature;
                fake_record.target_mass =
                    detector_model->GetTargetMass(target);
                double target_prob =
                    target_density
                    * cross_section->TotalCrossSection(fake_record);
                total_rate += target_prob;
                if (signature == record.signature) {
                    selected_rate += target_prob;
                }
            }
        }
    }

    if (candidate_count != nullptr) {
        *candidate_count = candidates;
    }
    return {total_rate, selected_rate};
}

// Find the FinalStateProbability for the matched interaction.
// Returns rate-weighted FinalStateProbability for the selected signature.
double RateWeightedFinalStateProbability(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record,
    PhaseSpaceConvention const & convention)
{
    siren::math::Vector3D interaction_vertex(
            record.interaction_vertex[0],
            record.interaction_vertex[1],
            record.interaction_vertex[2]);

    siren::math::Vector3D primary_direction(
            record.primary_momentum[1],
            record.primary_momentum[2],
            record.primary_momentum[3]);
    primary_direction.normalize();

    siren::geometry::Geometry::IntersectionList intersections_list =
        detector_model->GetIntersections(
            DetectorPosition(interaction_vertex),
            DetectorDirection(primary_direction));

    std::set<siren::dataclasses::ParticleType> const & possible_targets =
        interactions->TargetTypes();
    std::set<siren::dataclasses::ParticleType> available_targets =
        detector_model->GetAvailableTargets(
            intersections_list, DetectorPosition(record.interaction_vertex));

    double selected_final_state = 0.0;
    siren::dataclasses::InteractionRecord fake_record = record;

    // Decays
    for (auto const & decay : interactions->GetDecays()) {
        for (auto const & signature :
             decay->GetPossibleSignaturesFromParent(
                 record.signature.primary_type)) {
            if (signature == record.signature) {
                fake_record.signature = signature;
                double decay_prob = 1.0 / (
                    decay->TotalDecayLength(fake_record)
                    / siren::utilities::Constants::cm);
                double density = ConvertFinalStateDensity(
                    decay->FinalStateProbability(record),
                    decay->TopologyForSignature(signature),
                    decay->MeasureForSignature(signature),
                    convention, record);
                selected_final_state += decay_prob * density;
            }
        }
    }

    // Cross sections
    for (auto const target : available_targets) {
        if (possible_targets.find(target) == possible_targets.end())
            continue;
        double target_density =
            detector_model->GetParticleDensity(
                intersections_list,
                DetectorPosition(interaction_vertex), target);
        for (auto const & cross_section :
             interactions->GetCrossSectionsForTarget(target)) {
            for (auto const & signature :
                 cross_section->GetPossibleSignaturesFromParents(
                     record.signature.primary_type, target)) {
                if (signature == record.signature) {
                    fake_record.signature = signature;
                    fake_record.target_mass =
                        detector_model->GetTargetMass(target);
                    double target_prob =
                        target_density
                        * cross_section->TotalCrossSection(fake_record);
                    double density = ConvertFinalStateDensity(
                        cross_section->FinalStateProbability(record),
                        cross_section->TopologyForSignature(signature),
                        cross_section->MeasureForSignature(signature),
                        convention, record);
                    selected_final_state += target_prob * density;
                }
            }
        }
    }

    return selected_final_state;
}

} // anonymous namespace

PhaseSpaceConvention SelectedFinalStateConvention(
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record)
{
    return ResolveSelectedFinalStateConvention(interactions, record);
}

PhaseSpaceConvention ResolveCommonFinalStateConvention(
    PhaseSpaceConvention const & first,
    PhaseSpaceConvention const & second)
{
    if (first == second) return first;

    if (first.topology != second.topology) {
        std::ostringstream oss;
        oss << "Final-state densities have different topologies ("
            << siren::dataclasses::PhaseSpaceTopologyName(first.topology)
            << " and "
            << siren::dataclasses::PhaseSpaceTopologyName(second.topology)
            << "); there is no common pointwise convention "
            << "[siren-docs: errors#measure-compat]";
        throw siren::utilities::MeasureCompatibilityError(oss.str());
    }

    if (PhaseSpaceDensityConvertible(
            first.topology, first.measure, second.measure)) {
        return second;
    }
    if (PhaseSpaceDensityConvertible(
            first.topology, second.measure, first.measure)) {
        return first;
    }

    std::ostringstream oss;
    oss << "Final-state densities use incompatible measures ("
        << siren::dataclasses::PhaseSpaceMeasureName(first.measure)
        << " and "
        << siren::dataclasses::PhaseSpaceMeasureName(second.measure)
        << ") within "
        << siren::dataclasses::PhaseSpaceTopologyName(first.topology)
        << "; there is no common pointwise convention "
        << "[siren-docs: errors#measure-compat]";
    throw siren::utilities::MeasureCompatibilityError(oss.str());
}

double SelectedFinalStateProbability(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record)
{
    return SelectedFinalStateProbability(
        detector_model, interactions, record,
        ResolveSelectedFinalStateConvention(interactions, record));
}

double SelectedFinalStateProbability(
    std::shared_ptr<siren::detector::DetectorModel const>,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record,
    PhaseSpaceConvention const & convention)
{
    // Find the matching Decay or CrossSection and return its final-state
    // density in the requested convention (no rate weighting).
    if (interactions->HasDecays()) {
        for (auto const & decay : interactions->GetDecays()) {
            for (auto const & sig :
                 decay->GetPossibleSignaturesFromParent(
                     record.signature.primary_type)) {
                if (sig == record.signature) {
                    return ConvertFinalStateDensity(
                        decay->FinalStateProbability(record),
                        decay->TopologyForSignature(sig),
                        decay->MeasureForSignature(sig),
                        convention, record);
                }
            }
        }
    }
    if (interactions->HasCrossSections()) {
        for (auto const & xs_list : interactions->GetCrossSectionsByTarget()) {
            for (auto const & xs : xs_list.second) {
                for (auto const & sig :
                     xs->GetPossibleSignaturesFromParents(
                         record.signature.primary_type,
                         xs_list.first)) {
                    if (sig == record.signature) {
                        return ConvertFinalStateDensity(
                            xs->FinalStateProbability(record),
                            xs->TopologyForSignature(sig),
                            xs->MeasureForSignature(sig),
                            convention, record);
                    }
                }
            }
        }
    }
    return 0.0;
}

double ChannelSelectionProbability(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record)
{
    auto [total_rate, selected_rate] =
        AccumulateRates(detector_model, interactions, record);
    if (total_rate == 0) return 0.0;
    return selected_rate / total_rate;
}

double FixedVertexChannelSelectionProbability(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record)
{
    std::size_t candidate_count = 0;
    auto [total_rate, selected_rate] =
        AccumulateRates(detector_model, interactions, record, &candidate_count);

    // Exact float no-op when at most one channel competes: the injector's
    // rate selection was trivial, so no branching factor applies. Guarding on
    // the candidate count (not on selected_rate == total_rate) keeps existing
    // single-channel Fixed configs bit-for-bit unchanged even if the single
    // rate is zero or non-finite.
    if (candidate_count <= 1) {
        return 1.0;
    }

    // Multiple channels compete: fail loud rather than silently returning a
    // wrong factor.
    if (total_rate <= 0.0) {
        throw siren::utilities::WeightCalculationError(
            "FixedVertexChannelSelectionProbability: total interaction rate is "
            "non-positive (" + std::to_string(total_rate) + ") across "
            + std::to_string(candidate_count)
            + " competing channels; cannot form the channel-selection factor.");
    }
    double probability = selected_rate / total_rate;
    if (not std::isfinite(probability)) {
        throw siren::utilities::WeightCalculationError(
            "FixedVertexChannelSelectionProbability: non-finite channel-selection "
            "probability (selected_rate=" + std::to_string(selected_rate)
            + ", total_rate=" + std::to_string(total_rate) + ").");
    }
    return probability;
}

double CrossSectionProbability(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record)
{
    return CrossSectionProbability(
        detector_model, interactions, record,
        ResolveSelectedFinalStateConvention(interactions, record));
}

double CrossSectionProbability(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record,
    PhaseSpaceConvention const & convention)
{
    auto [total_rate, selected_rate] =
        AccumulateRates(detector_model, interactions, record);
    if (total_rate == 0) return 0.0;

    double selected_final_state =
        RateWeightedFinalStateProbability(
            detector_model, interactions, record, convention);

    return selected_final_state / total_rate;
}

double CrossSectionProbabilityWithPhaseSpace(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record,
    MultiChannelPhaseSpace const & phase_space)
{
    return CrossSectionProbabilityWithPhaseSpace(
        detector_model, interactions, record, phase_space,
        phase_space.CommonConvention());
}

double CrossSectionProbabilityWithPhaseSpace(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    siren::dataclasses::InteractionRecord const & record,
    MultiChannelPhaseSpace const & phase_space,
    PhaseSpaceConvention const & convention)
{
    auto [total_rate, selected_rate] =
        AccumulateRates(detector_model, interactions, record);
    if (total_rate == 0 || selected_rate == 0) return 0.0;

    double mc_density = phase_space.DensityIn(
        detector_model, record, convention);

    return (selected_rate * mc_density) / total_rate;
}

} // namespace injection
} // namespace siren

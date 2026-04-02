#include "SIREN/distributions/primary/vertex/BoxVolumePositionDistribution.h"

#include <cmath>
#include <vector>
#include <stdexcept>

#include "SIREN/dataclasses/InteractionRecord.h"
#include "SIREN/detector/DetectorModel.h"
#include "SIREN/geometry/Geometry.h"
#include "SIREN/math/Vector3D.h"
#include "SIREN/utilities/Random.h"

namespace siren {
namespace distributions {

// -------------------------------
// SamplePosition (BOX VERSION)
// -------------------------------
std::tuple<siren::math::Vector3D, siren::math::Vector3D>
BoxVolumePositionDistribution::SamplePosition(
    std::shared_ptr<siren::utilities::SIREN_random> rand,
    std::shared_ptr<siren::detector::DetectorModel const>,
    std::shared_ptr<siren::interactions::InteractionCollection const>,
    siren::dataclasses::PrimaryDistributionRecord & record) const {

    // --- Sample uniformly in box ---
    double xlen = box.GetX();
    double ylen = box.GetY();
    double zlen = box.GetZ();

    double xmin = -xlen/2.0;
    double xmax =  xlen/2.0;

    double ymin = -ylen/2.0;
    double ymax =  ylen/2.0;

    double zmin = -zlen/2.0;
    double zmax =  zlen/2.0;

    double x = rand->Uniform(xmin, xmax);
    double y = rand->Uniform(ymin, ymax);
    double z = rand->Uniform(zmin, zmax);

    siren::math::Vector3D local_pos(x, y, z);
    siren::math::Vector3D final_pos = box.LocalToGlobalPosition(local_pos);
    // --- Get direction ---
    siren::math::Vector3D dir = record.GetDirection();

    // --- Find entry point (same logic as cylinder) ---
    std::vector<siren::geometry::Geometry::Intersection> intersections =
        box.Intersections(final_pos, dir);

    siren::detector::DetectorModel::SortIntersections(intersections);

    siren::math::Vector3D init_pos;

    if (intersections.size() == 0) {
        init_pos = final_pos;
    } else if (intersections.size() >= 2) {
        init_pos = intersections.front().position;
    } else {
        throw std::runtime_error("Only found one box intersection!");
    }

    return {init_pos, final_pos};
}

// -------------------------------
// GenerationProbability
// -------------------------------
double BoxVolumePositionDistribution::GenerationProbability(
    std::shared_ptr<siren::detector::DetectorModel const>,
    std::shared_ptr<siren::interactions::InteractionCollection const>,
    siren::dataclasses::InteractionRecord const & record) const {

    siren::math::Vector3D pos(box.GlobalToLocalPosition(record.interaction_vertex));

    double xlen = box.GetX();
    double ylen = box.GetY();
    double zlen = box.GetZ();

    if (std::abs(pos.GetX()) > xlen/2.0 ||
        std::abs(pos.GetY()) > ylen/2.0 ||
    	std::abs(pos.GetZ()) > zlen/2.0) {
        return 0.0;
    }

    double volume = box.GetX() * box.GetY() * box.GetZ();
    return 1.0 / volume;
}

// -------------------------------
BoxVolumePositionDistribution::BoxVolumePositionDistribution(siren::geometry::Box b)
    : box(b) {}

// -------------------------------
std::string BoxVolumePositionDistribution::Name() const {
    return "BoxVolumePositionDistribution";
}

// -------------------------------
std::shared_ptr<PrimaryInjectionDistribution>
BoxVolumePositionDistribution::clone() const {
    return std::shared_ptr<PrimaryInjectionDistribution>(
        new BoxVolumePositionDistribution(*this));
}

// -------------------------------
std::tuple<siren::math::Vector3D, siren::math::Vector3D>
BoxVolumePositionDistribution::InjectionBounds(
    std::shared_ptr<siren::detector::DetectorModel const>,
    std::shared_ptr<siren::interactions::InteractionCollection const>,
    siren::dataclasses::InteractionRecord const & interaction) const {

    siren::math::Vector3D dir(
        interaction.primary_momentum[1],
        interaction.primary_momentum[2],
        interaction.primary_momentum[3]);

    dir.normalize();

    siren::math::Vector3D pos(interaction.interaction_vertex);

    std::vector<siren::geometry::Geometry::Intersection> intersections =
        box.Intersections(pos, dir);

    siren::detector::DetectorModel::SortIntersections(intersections);

    if (intersections.size() == 0) {
        return {siren::math::Vector3D(0,0,0), siren::math::Vector3D(0,0,0)};
    } else if (intersections.size() >= 2) {
        return {intersections.front().position, intersections.back().position};
    } else {
        throw std::runtime_error("Only found one box intersection!");
    }
}

// -------------------------------
bool BoxVolumePositionDistribution::less(WeightableDistribution const & other) const {
    const BoxVolumePositionDistribution* x =
        dynamic_cast<const BoxVolumePositionDistribution*>(&other);

    if(!x) return false;
    return box < x->box;
}

// -------------------------------
bool BoxVolumePositionDistribution::AreEquivalent(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    std::shared_ptr<WeightableDistribution const> distribution,
    std::shared_ptr<siren::detector::DetectorModel const> second_detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> second_interactions
) const {
    return this->operator==(*distribution);
}

bool BoxVolumePositionDistribution::equal(WeightableDistribution const & other) const {
    const BoxVolumePositionDistribution* x =
        dynamic_cast<const BoxVolumePositionDistribution*>(&other);

    if(!x) return false;
    return (box == x->box);
}
} // namespace distributions
} // namespace siren


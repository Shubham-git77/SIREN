#include "SIREN/distributions/primary/PrimaryExternalDistribution.h"

#include <algorithm>                                       // for min
#include <array>                                           // for array
#include <cmath>                                           // for sqrt
#include <fstream>                                         // for ifstream
#include <sstream>                                         // for stringstream
#include <string>                                          // for basic_string
#include <stdexcept>                                       // for runtime_error
#include <tuple>                                           // for tie

#include "SIREN/dataclasses/InteractionRecord.h"  // for InteractionRecord
#include "SIREN/utilities/Random.h"               // for SIREN_random

namespace siren {
namespace distributions {

//---------------
// class PrimaryExternalDistribution
//---------------

static std::string trim(std::string const & s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void PrimaryExternalDistribution::LoadInputFile(std::string const & _filename) {
    filename = _filename;
    keys.clear();
    input_data.clear();
    init_pos_set = false;
    mom_set = false;

    std::ifstream input_file(filename);
    if (!input_file.is_open()) {
        throw std::runtime_error("error: file open failed " + filename);
    }

    std::string line;
    std::getline(input_file, line);

    std::stringstream ss(line);
    std::string key;
    bool has_x0 = false, has_y0 = false, has_z0 = false;
    bool has_px = false, has_py = false, has_pz = false;
    while (std::getline(ss, key, ',')) {
        key = trim(key);
        keys.push_back(key);
        if      (key == "x0") has_x0 = true;
        else if (key == "y0") has_y0 = true;
        else if (key == "z0") has_z0 = true;
        else if (key == "px") has_px = true;
        else if (key == "py") has_py = true;
        else if (key == "pz") has_pz = true;
    }
    init_pos_set = has_x0 && has_y0 && has_z0;
    mom_set      = has_px && has_py && has_pz;

    std::string value;
    while (std::getline(input_file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        std::vector<double> tmp_data;
        std::stringstream _ss(line);
        size_t ikey = 0;
        bool passed = true;
        while (std::getline(_ss, value, ',')) {
            if (ikey >= keys.size()) {
                throw std::runtime_error("CSV row has more columns than header in " + filename);
            }
            if (keys[ikey] == "E") {
                if (stod(value) < emin) passed = false;
            }
            ++ikey;
            tmp_data.push_back(stod(value));
        }
        if (ikey != keys.size()) {
            throw std::runtime_error("CSV row has fewer columns than header in " + filename);
        }
        if (passed) input_data.push_back(tmp_data);
    }

    if (input_data.empty()) {
        throw std::runtime_error("No valid data rows in " + filename);
    }
}

PrimaryExternalDistribution::PrimaryExternalDistribution(std::string _filename)
    : emin(0) {
    LoadInputFile(_filename);
}

PrimaryExternalDistribution::PrimaryExternalDistribution(std::string _filename, double emin)
    : emin(emin) {
    LoadInputFile(_filename);
}

// Accounts for events above threshold only!
size_t PrimaryExternalDistribution::GetPhysicalNumEvents() const {
    return input_data.size();
}

void PrimaryExternalDistribution::Sample(
        std::shared_ptr<siren::utilities::SIREN_random> rand,
        std::shared_ptr<siren::detector::DetectorModel const> detector_model,
        std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
        siren::dataclasses::PrimaryDistributionRecord & record) const {

    size_t i = std::min(size_t(rand->Uniform() * input_data.size()),
                        input_data.size() - 1);

    std::array<double, 3> _initial_position = {0.0, 0.0, 0.0};
    std::array<double, 3> _momentum         = {0.0, 0.0, 0.0};

    for (size_t i_key = 0; i_key < keys.size(); ++i_key) {
        double value = input_data[i][i_key];
        if      (keys[i_key] == "x0") { _initial_position[0] = value; }
        else if (keys[i_key] == "y0") { _initial_position[1] = value; }
        else if (keys[i_key] == "z0") { _initial_position[2] = value; }
        else if (keys[i_key] == "px") { _momentum[0] = value; }
        else if (keys[i_key] == "py") { _momentum[1] = value; }
        else if (keys[i_key] == "pz") { _momentum[2] = value; }
        else if (keys[i_key] == "E")  { record.SetEnergy(value); }
        else if (keys[i_key] == "m")  { record.SetMass(value); }
        else { record.SetInteractionParameter(keys[i_key], value); }
    }

    if (mom_set)      record.SetThreeMomentum(_momentum);
    if (init_pos_set) {
        // Set BOTH the initial position (start of track) and the interaction
        // vertex to the CSV decay point (x0, y0, z0).  The vertex is where
        // the pion actually decayed — it must come from the file, not be
        // re-derived from geometry.
        record.SetInitialPosition(_initial_position);
        record.SetInteractionVertex(_initial_position);

        // Cache for SamplePosition() — SIREN still calls this as part of the
        // VertexPositionDistribution interface, so we must return a valid
        // non-degenerate segment.  We build it from the CSV values so that
        // even the geometric fallback is grounded in the file data.
        _cached_position = _initial_position;
    }
    if (mom_set) {
        double pmag = std::sqrt(_momentum[0]*_momentum[0]
                               + _momentum[1]*_momentum[1]
                               + _momentum[2]*_momentum[2]);
        _cached_direction = (pmag > 0.0)
            ? std::array<double,3>{ _momentum[0]/pmag,
                                    _momentum[1]/pmag,
                                    _momentum[2]/pmag }
            : std::array<double,3>{ 0.0, 0.0, 1.0 };
    }
    _cache_valid = init_pos_set;
}

// VertexPositionDistribution implementation.
// The true vertex is already fixed in Sample() via SetInteractionVertex().
// SamplePosition() is still required by the interface; it returns a tiny
// segment centred on the CSV decay point so the injector has a valid
// direction and non-zero length if it needs them.
std::tuple<siren::math::Vector3D, siren::math::Vector3D>
PrimaryExternalDistribution::SamplePosition(
    std::shared_ptr<siren::utilities::SIREN_random> /*rand*/,
    std::shared_ptr<siren::detector::DetectorModel const> /*detector_model*/,
    std::shared_ptr<siren::interactions::InteractionCollection const> /*interactions*/,
    siren::dataclasses::PrimaryDistributionRecord & /*record*/) const
{
    if (_cache_valid) {
        siren::math::Vector3D pos(_cached_position[0],
                                  _cached_position[1],
                                  _cached_position[2]);
        siren::math::Vector3D dir(_cached_direction[0],
                                  _cached_direction[1],
                                  _cached_direction[2]);
        // Tiny segment centred on the CSV decay point.
        constexpr double half_len = 1e-4;
        return { pos - half_len * dir,
                 pos + half_len * dir };
    }
    // Fallback — should never be reached when the CSV has x0/y0/z0.
    return { siren::math::Vector3D(0, 0, 0),
             siren::math::Vector3D(0, 0, 1e-4) };
}

std::tuple<siren::math::Vector3D, siren::math::Vector3D>
PrimaryExternalDistribution::InjectionBounds(
    std::shared_ptr<siren::detector::DetectorModel const> /*detector_model*/,
    std::shared_ptr<siren::interactions::InteractionCollection const> /*interactions*/,
    siren::dataclasses::InteractionRecord const & /*record*/) const
{
    // No analytic bounds — positions come entirely from the CSV.
    return { siren::math::Vector3D(0, 0, 0),
             siren::math::Vector3D(0, 0, 0) };
}

std::vector<std::string> PrimaryExternalDistribution::DensityVariables() const {
    return std::vector<std::string>{"External", "Position"};
}

std::string PrimaryExternalDistribution::Name() const {
    return "PrimaryExternalDistribution";
}

double PrimaryExternalDistribution::GenerationProbability(
        std::shared_ptr<siren::detector::DetectorModel const> /*detector_model*/,
        std::shared_ptr<siren::interactions::InteractionCollection const> /*interactions*/,
        siren::dataclasses::InteractionRecord const & record) const {
    double energy = record.primary_momentum[0];
    return (energy >= emin) ? 1.0 : 0.0;
}

std::shared_ptr<PrimaryInjectionDistribution> PrimaryExternalDistribution::clone() const {
    return std::shared_ptr<PrimaryInjectionDistribution>(
        new PrimaryExternalDistribution(*this));
}

bool PrimaryExternalDistribution::equal(WeightableDistribution const & other) const {
    const PrimaryExternalDistribution* x =
        dynamic_cast<const PrimaryExternalDistribution*>(&other);
    if (!x) return false;
    return emin       == x->emin
        && keys       == x->keys
        && input_data == x->input_data;
}

bool PrimaryExternalDistribution::less(WeightableDistribution const & other) const {
    const PrimaryExternalDistribution* x =
        dynamic_cast<const PrimaryExternalDistribution*>(&other);
    if (!x) return false;
    return std::tie(emin, keys, input_data)
         < std::tie(x->emin, x->keys, x->input_data);
}

} // namespace distributions
} // namespace siren

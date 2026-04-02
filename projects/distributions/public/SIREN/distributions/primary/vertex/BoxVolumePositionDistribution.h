#pragma once
#ifndef SIREN_BoxVolumePositionDistribution_H
#define SIREN_BoxVolumePositionDistribution_H

#include <tuple>
#include <memory>
#include <string>
#include <cstdint>
#include <stdexcept>

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/utility.hpp>

#include "SIREN/distributions/primary/vertex/VertexPositionDistribution.h"
#include "SIREN/geometry/Box.h"
#include "SIREN/math/Vector3D.h"

namespace siren { namespace interactions { class InteractionCollection; } }
namespace siren { namespace dataclasses { class InteractionRecord; } }
namespace siren { namespace detector { class DetectorModel; } }
namespace siren { namespace distributions { class PrimaryInjectionDistribution; } }
namespace siren { namespace distributions { class WeightableDistribution; } }
namespace siren { namespace utilities { class SIREN_random; } }

namespace siren {
namespace distributions {

class BoxVolumePositionDistribution : virtual public VertexPositionDistribution {
friend cereal::access;
protected:
    BoxVolumePositionDistribution() {};
private:
    siren::geometry::Box box;

    std::tuple<siren::math::Vector3D, siren::math::Vector3D>
    SamplePosition(std::shared_ptr<siren::utilities::SIREN_random> rand,
                   std::shared_ptr<siren::detector::DetectorModel const>,
                   std::shared_ptr<siren::interactions::InteractionCollection const>,
                   siren::dataclasses::PrimaryDistributionRecord &) const override;

public:
    virtual double GenerationProbability(std::shared_ptr<siren::detector::DetectorModel const>,
                                         std::shared_ptr<siren::interactions::InteractionCollection const>,
                                         siren::dataclasses::InteractionRecord const &) const override;

    BoxVolumePositionDistribution(siren::geometry::Box);

    std::string Name() const override;
    virtual std::shared_ptr<PrimaryInjectionDistribution> clone() const override;

    virtual std::tuple<siren::math::Vector3D, siren::math::Vector3D>
    InjectionBounds(std::shared_ptr<siren::detector::DetectorModel const>,
                    std::shared_ptr<siren::interactions::InteractionCollection const>,
                    siren::dataclasses::InteractionRecord const &) const override;

virtual bool AreEquivalent(
    std::shared_ptr<siren::detector::DetectorModel const> detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> interactions,
    std::shared_ptr<WeightableDistribution const> distribution,
    std::shared_ptr<siren::detector::DetectorModel const> second_detector_model,
    std::shared_ptr<siren::interactions::InteractionCollection const> second_interactions
) const override;

protected:
    virtual bool equal(WeightableDistribution const &) const override;
    virtual bool less(WeightableDistribution const &) const override;
};

} // namespace distributions
} // namespace siren

CEREAL_REGISTER_TYPE(siren::distributions::BoxVolumePositionDistribution);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    siren::distributions::VertexPositionDistribution,
    siren::distributions::BoxVolumePositionDistribution
);

#endif


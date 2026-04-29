#include <gtest/gtest.h>

#include "SIREN/distributions/Distributions.h"
#include "SIREN/distributions/primary/energy/Monoenergetic.h"
#include "SIREN/distributions/primary/energy/PowerLaw.h"
#include "SIREN/distributions/primary/direction/IsotropicDirection.h"

using namespace siren::distributions;

TEST(WeightableDistribution, OperatorLessSameType) {
    Monoenergetic a(10.0);
    Monoenergetic b(20.0);
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
}

TEST(WeightableDistribution, StrictWeakOrderingSameType) {
    Monoenergetic a(10.0);
    Monoenergetic b(20.0);
    Monoenergetic c(30.0);
    // Irreflexivity
    EXPECT_FALSE(a < a);
    // Asymmetry
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    // Transitivity
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b < c);
    EXPECT_TRUE(a < c);
}

TEST(WeightableDistribution, OperatorEqualSameType) {
    Monoenergetic a(10.0);
    Monoenergetic b(10.0);
    Monoenergetic c(20.0);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(WeightableDistribution, OperatorEqualDifferentTypes) {
    Monoenergetic mono(10.0);
    PowerLaw power(2.0, 1.0, 100.0);
    IsotropicDirection iso;
    EXPECT_FALSE(mono == power);
    EXPECT_FALSE(mono == iso);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

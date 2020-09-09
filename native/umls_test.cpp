#include "umls.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

class UMLSTest : public ::testing::Test {
   public:
    UMLS umls;
};

TEST_F(UMLSTest, GetAui) {
    EXPECT_EQ(umls.get_aui("ICD10CM", "E11"), "A17825389");
}
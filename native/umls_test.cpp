#include "umls.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

class UMLSTest : public ::testing::Test {
    
   public:
    UMLSTest() : umls("/share/pi/nigam/ethanid/UMLS") {

    }
    UMLS umls;
};

// TEST_F(UMLSTest, GetAui) {
//     EXPECT_EQ(umls.get_aui("ICD10CM", "E11"), "A17825389");
// }

TEST_F(UMLSTest, TestCycle) {
    auto aui = umls.get_aui("LNC", "LP343631-0");

    std::cout<<"Help" << aui.has_value() << std::endl;
    std::cout<<"Blah" << aui.value() << std::endl;

    for (const auto& a : umls.get_parents(*aui)) {
        std::cout<<"Got parent " << a << std::endl;
        auto details = *umls.get_code(a);

        std::cout<<"What " << details.first << " " << details.second << std::endl;
    }

}
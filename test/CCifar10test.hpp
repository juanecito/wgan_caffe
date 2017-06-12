/*
 * CCifar10test.hpp
 *
 *  Created on: 12 jun. 2017
 *      Author: juan
 */

#ifndef TEST_CCIFAR10TEST_HPP_
#define TEST_CCIFAR10TEST_HPP_

// apt install libgtest-dev
#include "gtest/gtest.h"

class CCifar10_test: public testing::Test
{
public:
	CCifar10_test( ) {
       // initialization code here
   }

   void SetUp( ) {
       // code here will execute just before the test ensues
   }

   void TearDown( ) {
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }

   ~CCifar10_test( )  {
       // cleanup any pending stuff, but no exceptions allowed
   }

   // put in any custom data members that you need
};


#endif /* TEST_CCIFAR10TEST_HPP_ */

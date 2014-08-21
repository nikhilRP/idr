/*****************************************************************************
  main.cpp

  (c) 2014 - Nikhil R Podduturi
  J. Michael Cherry Lab, Department of Genetics, Stanford University School of Medicine

  Licensed under the GNU General Public License 2.0 license.
******************************************************************************/

// gtest based unit tests
#include ""

#include "command-line-args-test.h"

TIMED_SCOPE(testTimer, "IDR Unit Tests");

int main() {
    testing::InitGoogleTest(&argc, argv);
    _INIT_SYSLOG(kSysLogIdent, 0, 0);

    reconfigureLoggersForTest();
    std::cout << "Logs for test are written in [" << logfile << "]" << std::endl;

    return ::testing::UnitTest::GetInstance()->Run();
}

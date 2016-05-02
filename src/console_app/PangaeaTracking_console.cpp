#include "main_engine/MainEngine.h"
#if defined(_DEBUG) && defined(_MSC_VER)
#include "vld.h"
#endif
int main(int argc, char* argv[])
{
  MainEngine mainEngine;
  mainEngine.ReadConfigurationFile(argc, argv);
  mainEngine.SetupInputAndTracker();
  mainEngine.Run();

  return 0;

}

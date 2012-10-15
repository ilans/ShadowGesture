// ShadowGesture.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "ShadowGesture.h"

int _tmain(int argc, _TCHAR* argv[])
{
	ShadowGesture* sg = new ShadowGesture();
	//sg->capture();
	//sg->extract();
	sg->vectorize();
	sg->trainHMM();
	sg->testHMM("../Data/test_sequense.yml");
	//sg->FindConvexityDefects("../Data/images/hand126_0.png");
	
	return 0;
}


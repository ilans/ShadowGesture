// ShadowGesture.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "ShadowGesture.h"

int _tmain(int argc, _TCHAR* argv[])
{
	ShadowGesture* sg = new ShadowGesture();
	//sg->capture("../Data/making_gestures_for_testing1.m4v");
	//sg->extract();
	//sg->vectorize();
	//sg->trainHMM();
	//sg->testHMM("../Data/test_sequense.yml");
	//sg->FindConvexityDefects("../Data/images/hand126_0.png");
	//sg->recognizeGesture("../Data/making_gestures_for_testing1.m4v");
	//sg->convertDataToOctaveCVS("../Data/train_sequense.yml");
	sg->convertDataToOctaveCVS("../Data/test_sequense.yml");

	return 0;
}


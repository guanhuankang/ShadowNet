from demo import Demo

imgPath = "../dataset/SBU-shadow/SBU-Test/ShadowImages"
gtPath = "../dataset/SBU-shadow/SBU-Test/ShadowMasks"

# imgPath = "../dataset/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages"
# gtPath = "../dataset/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks"

name = "SBU_5000"
demo = Demo(modelPath="models/{}.pt".format(name))
demo.eval(imgPath=imgPath, outPath=name, gtPath=gtPath, name=name)

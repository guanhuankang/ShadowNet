from demo import Demo

imgPath = "../dataset/SBU-shadow/SBU-Test/ShadowImages"
gtPath = "../dataset/SBU-shadow/SBU-Test/ShadowMasks"

name = "SBU_2000"
demo = Demo(modelPath="models/{}.pt".format(name))
demo.eval(imgPath=imgPath, outPath=name, gtPath=gtPath, name=name)

from demo import Demo

imgPath = "../dataset/SBU-shadow/SBU-Test/ShadowImages"
gtPath = "../dataset/SBU-shadow/SBU-Test/ShadowMasks"

demo = Demo(modelPath="models/SBU_5000.pt")
demo.eval(imgPath=imgPath, outPath="results5000", gtPath=gtPath, name="iter5000")

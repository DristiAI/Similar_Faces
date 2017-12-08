from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import imutils
import cv2
print("extracting features")
data=[]
labels=[]
for imagepath in paths.list_images("database"):
    make=imagepath.split("/")[-2]
    image=cv2.imread(imagepath)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #print(make)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

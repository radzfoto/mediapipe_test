import pkgutil
import mediapipe.python.solutions as solutions
from mediapipe.tasks.python import vision

a = vision.FaceLandmarker

for importer, modname, ispkg in pkgutil.iter_modules(solutions.__path__, solutions.__name__ + "."):
    print(modname)

for importer, modname, ispkg in pkgutil.iter_modules(vision.__path__, vision.__name__ + "."):
    print(modname)

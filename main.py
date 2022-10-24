import face_mesh_generator as f_mesh
import os
from os import listdir
import face_net_test as f_net


if __name__ == "__main__":

    # Obtaining face mesh
    FILEPATH = "data/utk_dataset/UTKFace"
    # for images in os.listdir(FILEPATH):
    #     f_mesh.obtain_face_mesh(FILEPATH,images)
    f_net.create_dataframe(FILEPATH)


    #build CNN and test 

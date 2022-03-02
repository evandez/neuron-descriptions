MANUAL_CHECK = \
"""\nManual check... 
  Check that the error above is like "Couldn't find any class folder in".
  This is probably because your imagenet val folder is not arranged by class.

  ################################################
  Assume your the imagenet folder is like
    imagenet/
      (1) train (2) val (3) test 
      presumably copied from the original ILSVRC/Data/CLS-LOC folder.
  Then copy the original ILSVRC/Annotations/CLS-LOC/val into imagenet/Annotations/val 
  before continuing. This is for the script to find validation image labels.
  ################################################


  Input "OK" (without quotation, ALL CAPITAL LETTERS) to create an alternative folder 
  named "val_rearranged".
  Otherwise, input everything else.
Your input here:
"""

import os, shutil
from xml.dom.minidom import parse
from .imagenet_dict2 import ID2LABELS
def try_alternative_val_folder(source_dir, err):
    """
    source_dir: dir of data
    err: error caught

    ############### IMPORTANT! ##################
    ASSUME you have created the following data folder: 
    $MILAN_DATA_DIR/imagenet/val
    where val is the problematic folder that is not arranged in folders by class.
    """
    
    print(err,'\nAttempting to use rearranged val folder...')
    IMAGENET_FOLDER_DIR = os.path.join(*source_dir.parts[:-1])
    alternative_source_dir = os.path.join(IMAGENET_FOLDER_DIR,'val_rearranged')  

    if not os.path.exists(alternative_source_dir):
        userconfirmation = input(MANUAL_CHECK)
        if userconfirmation == 'OK':
            create_alternative_folder(IMAGENET_FOLDER_DIR, source_dir, alternative_source_dir)
        else:
            print('not proceeding with error.')
            exit()
    else:
        print('alternative val folder already exists. Attempting to use it.')
    return alternative_source_dir

def create_alternative_folder(IMAGENET_FOLDER_DIR, source_dir, alternative_source_dir):
    print('creating alternative folder %s...'%(str(alternative_source_dir)))
    os.makedirs( alternative_source_dir ,exist_ok=True)
    imgnetval = ImageNetValidation(IMAGENET_FOLDER_DIR)

    n_val = 50000
    for i in range(n_val):
        img_name, this_index, label_text = imgnetval.get_val_data_by_index(i)
        if i<10:
            print(img_name, this_index, label_text) 
        elif i==10:
            print('...\n')
        else:
            if (i+1)%10==0 or (i+1)==n_val:
                print('%s/%s'%(str(i+1),str(n_val)), end='\r')
        subfolder = os.path.join(alternative_source_dir, str(this_index))

        os.makedirs(subfolder, exist_ok=True )
        shutil.copyfile( os.path.join(imgnetval.MAIN_DATA_BRANCH_DIR, img_name) , 
            os.path.join(subfolder,img_name))
    print('\nalternative val folder done!')


class ImageNetValidation():
    def __init__(self, IMAGENET_FOLDER_DIR):
        super(ImageNetValidation, self).__init__()
        

        self.IMAGENET_FOLDER_DIR = IMAGENET_FOLDER_DIR
        self.MAIN_DATA_BRANCH_DIR = os.path.join(IMAGENET_FOLDER_DIR, 'val')
        self.MAIN_VAL_LABELS_DIR = os.path.join(IMAGENET_FOLDER_DIR,'Annotations', 'val')
        self.VAL_IMG_LIST = os.listdir(self.MAIN_DATA_BRANCH_DIR)

    def get_val_data_by_index(self, i):
        assert(1<=i+1<=50000)
        img_xml_name = self.VAL_IMG_LIST[i].split('.')[0] + '.xml'
        img_name = self.VAL_IMG_LIST[i].split('.')[0] + '.JPEG'

        xml_dir = os.path.join(self.MAIN_VAL_LABELS_DIR,img_xml_name)
        img_dir = os.path.join(self.MAIN_DATA_BRANCH_DIR,img_name)

        xml = parse(xml_dir)
        this_node_list = xml.getElementsByTagName('name')
        this_node_name = this_node_list[0].childNodes[0].nodeValue
        try:
            this_index, label_text = ID2LABELS[this_node_name]
        except:
            return None, None, None

        return img_name, this_index, label_text

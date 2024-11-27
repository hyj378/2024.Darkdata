### 가공된 데이터를 통해 학습할 레퍼런스 정보 파일을 생성하는 코드입니다. 
import os
from glob import glob
import random

def write_file(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line+'\n')
    
if __name__ == '__main__':
    random.seed(42)
    savetrainpath = '../data/JB_data/trainval.txt' # 전체 학습 데이터의 레퍼런스가 저장될 경로
    savetestpath = '../data/JB_data/test.txt' # 전체 평가 데이터의 레퍼런스가 저장될 경로
    datapath = '../data/JB_data/xml/*' # Annotation 파일(xml)이 저장되어 있는 경로
    save_dir = '../data/few_shot_ann_JB/voc/' # Few-shot 레퍼런스가 저장될 폴더의 경로
    now_abspath = os.path.dirname(os.path.abspath(__file__)) 

    dirs = glob(datapath)
    train_lines = []
    test_lines = []
    for dir in dirs:
        filenames = os.listdir(dir)
        for filename in filenames:
            filename = os.path.join(now_abspath, dir, filename)
            filename = filename[:-3].replace('xml', 'frames')+'png'
            # "정전기앞" 데이터셋을 평가 데이터로 분류 
            if 'frames_static_pressure_machine' in dir: 
                test_lines.append(filename + " 0") # add zero because the image has only one object per a frame.
            else:
                train_lines.append(filename + " 0")

    ## Make Test Files Info ("test.txt")
    write_file(savetestpath, test_lines)

    ## Make Base Files Info ("trainval.txt")
    write_file(savetrainpath, train_lines)

    ### Make Random Few shot Info of Novel(Crane) ("benchmark_{x}shot/box_{x}shot_{class}_train.txt")
    save_path_temp = 'benchmark_{}shot/box_{}shot_{}_train.txt'
    class_name = 'crane'
    for k in [1,2,3,5,10]:
        if not os.path.exists(os.path.join(save_dir, f'benchmark_{k}shot')):
            os.makedirs(os.path.join(save_dir, f'benchmark_{k}shot'))
        random.shuffle(train_lines)
        few_selected = train_lines[:k]
        save_path = os.path.join(save_dir, save_path_temp.format(k, k, class_name))
        write_file(save_path, few_selected)
        

        
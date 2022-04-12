

import sys
import os
import math

import cv2 
import numpy as np
import argparse
import json
import csv
import copy
import glob


# ROMP path
rompRootPath = os.path.join(os.getenv("COMMONPROJECT"), '3rdparty_python/pose_estimation/video/ROMP/21.12.0/ROMP')
sys.path.append(rompRootPath)
rompModelPath = os.path.join(rompRootPath,'trained_models/ROMP_HRNet32_V1_ft_3DPW.pkl')
savedArgs = copy.copy(sys.argv)
	
# create args to be consumed by ROMP
sys.argv = ['romp','--model_path',rompModelPath]
from romp.predict.image import Single_image_predictor, instantiate_single_predictor
from romp.lib.visualization.visualization import draw_skeleton
import romp.lib.constants as constants
# restore args
sys.argv = savedArgs


def parse_args():
    parser = argparse.ArgumentParser()
	
    parser.add_argument('--personjsonfile', type=str, default=None,
                        help='Path to the COCO format JSON file with person detections to process')
						
    parser.add_argument('--personjsonfilepattern', type=str, default=None,
                        help='Path to the COCO format JSON fils with person detections to process, will process all files matching the pattern')
					
    parser.add_argument('--imgbasepath',type=str,default='',
                        help='Base path of images')

    parser.add_argument('--outdir',type=str,default='.',
                        help='Output directory')

    parser.add_argument('--showimg',action='store_true',default=False,
                        help='Show pose detection result images')
						
    parser.add_argument('--writeimg',action='store_true',default=False,
                        help='Write pose detection result images to ./skeletons')
						
    args = parser.parse_args()
    return args

def writeCSV(filename,posedict):

    header=['image']
    for ptname in constants.SMPL_ALL_54.keys():
        header.append(ptname+'_x')
        header.append(ptname+'_y')

    data = []
	
    for img in posedict.keys():
        row = [img]
        row.extend(posedict[img])
        if len(data)==0:
            data = [row]
        else:
            data.append(row)

    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
	
def drawPose(img,points):

    draw_skeleton(img, points, bones=constants.All54_connMat, cm=constants.cm_All54)
	

def processFile(anno,args):
     
	
    poseEstimator = instantiate_single_predictor()
	
    posedict = {}
	
    for a in anno['annotations']:
	
        iid = a['image_id']
        i = 0
        img = anno['images'][i]
        while not(img['id']==int(iid)):
            img = anno['images'][i]
            i = i+1
		
        print('processing '+img['file_name'])
		
        inputImgOrig = cv2.imread(args.imgbasepath+'/'+img['file_name'])	        
		
        bbox = a['bbox']
        # enlarge box
        fact = 0.2
        bbox[0] = bbox[0] - (bbox[2]*fact*0.5)
        bbox[1] = bbox[1] - (bbox[3]*fact*0.5)
        bbox[2] = bbox[2]*(1+fact)
        bbox[3] = bbox[3]*(1+fact)
		
        x1 = int(max(0,bbox[0]))
        y1 = int(max(0,bbox[1]))
        x2 = int(min(inputImgOrig.shape[1]-1,bbox[0]+bbox[2]))
        y2 = int(min(inputImgOrig.shape[0]-1,bbox[1]+bbox[3]))
		
        subImg = np.copy(inputImgOrig[y1:y2,x1:x2, : ])
		
        if subImg.shape[0]==0 or subImg.shape[1]==0:
            continue
		
        poseResult = poseEstimator.run(subImg)
		
        if '0' in poseResult.keys():
            if len(poseResult['0'])>0:
                if 'pj2d_org' in poseResult['0'][0].keys():
                    posePoints = poseResult['0'][0]['pj2d_org']
                else:
                    continue
            else:
                continue
        else:
            continue
			
        # shift to full image
        posePoints[:,0] = posePoints[:,0]+x1
        posePoints[:,1] = posePoints[:,1]+y1
		
        posedict[img['file_name']] = posePoints.flatten().astype(np.int32)
		
        if args.showimg or args.writeimg:
                
            #draw_skeleton(inputImgOrig, posePoints, bones=constants.All54_connMat, cm=constants.cm_All54)
            draw_skeleton(inputImgOrig, posePoints, bones=constants.smpl24_connMat, cm=constants.cm_smpl24)
				
            if args.showimg:
                cv2.imshow("skeletons", inputImgOrig )
                cv2.waitKey()
				
            if args.writeimg:
                cv2.imwrite('./skeletons/'+img['file_name'],inputImgOrig)

    return posedict
	
def main():
    args = parse_args()

	
    if args.personjsonfile is not None:
	
        f = open(args.personjsonfile)
        anno = json.load(f)   
	
        facedict = processFile(anno,args)
	
        if args.outdir is not None:
            writeCSV(args.outdir+'/poses.csv',facedict)
			
    elif args.personjsonfilepattern is not None:
	
        files = glob.glob(args.personjsonfilepattern)
		
        for filename in files:
            f = open(filename)
            anno = json.load(f)   
	
            posedict = processFile(anno,args)
	
            if args.outdir is not None:
                namepart = filename.split('/')[-1]
                namepart = namepart.split('\\')[-1]
                namepart = '.'.join(namepart.split('.')[:-1])
                writeCSV(args.outdir+'/poses_'+namepart+'.csv',posedict)	

if __name__ == "__main__":
    main()


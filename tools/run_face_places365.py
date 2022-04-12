

import sys
import os
import math

import cv2 
import numpy as np
import argparse
import json
import csv
import glob


# import RetinaFace
retinaFacePath = os.environ['COMMONPROJECT']  + '/3rdparty_python/face/detect/insightface/2021.01.13/detection/RetinaFace'
sys.path.append(retinaFacePath)

from api.FaceDetection import load_model, detect_faces_in_images

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
                        help='Write pose detection result images to ./faces')
						
    args = parser.parse_args()
    return args

def writeCSV(filename,posedict):

    header=['image','x','y','w','h']

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
	
def getAreaR2inR1(r1,r2):
   
    if r2[2] <= r1[0] or r2[3] <= r1[1] or r2[0] >= r1[2] or r2[1] >= r1[3]:
        return 0
		
    l = max(r1[0],r2[0])
    t = max(r1[1],r2[1])
    r = min(r1[2],r2[2])
    b = min(r1[3],r2[3])
	
    return (r-l)*(b-t)
	
	
def processFile(anno,args):

    faceModelHandle = load_model(retinaFacePath+'/model', 'JRS_R50_estimVisibFloat_AngularC')
	
    facedict = {}
	
    for a in anno['annotations']:
	
        iid = a['image_id']
        i = 0
        img = anno['images'][i]
        while not(img['id']==int(iid)):
            img = anno['images'][i]
            i = i+1
							
        inputImgOrig = cv2.imread(args.imgbasepath+'/'+img['file_name'])	
		
        imageList=[]
        img1 = {}
        img1['image_id'] = img['file_name']
        img1['image_data'] = inputImgOrig
        imageList.append(img1)
		
		
        resultList = detect_faces_in_images(imageList, faceModelHandle, threshold = 0.3)
		
        # for dbg
        overlaps = []
		
        for j, result in enumerate(resultList):
            maxoverlap = 0
            bestBbox = []
            if not('regions' in result.keys()):
                continue
            if len(result['regions'])>0:
                for i in range(len(result['regions'])):
                    bbox = [ int(result['regions'][i]['box']['tl_x']*inputImgOrig.shape[1]),
                        int(result['regions'][i]['box']['tl_y']*inputImgOrig.shape[0]),
                        int((result['regions'][i]['box']['br_x']-result['regions'][i]['box']['tl_x'])*inputImgOrig.shape[1]),
                        int((result['regions'][i]['box']['br_y']-result['regions'][i]['box']['tl_y'])*inputImgOrig.shape[0]) ]
 	   	 
                    overlap = getAreaR2inR1([a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]], [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
                    print(bbox)
                    print(overlap)
                    score = overlap * result['regions'][i]['confidence']
                    print(score)
                    if score > maxoverlap:
                        bestBbox = bbox
                        maxoverlap = score
                    overlaps.append(score)
		
        if len(bestBbox)>0:
            facedict[img['file_name']] = bestBbox
		
        if args.showimg or args.writeimg:
        
            color = (255,255,255)
            inputImgOrig = cv2.rectangle(inputImgOrig, [a['bbox'][0],a['bbox'][1]], [a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]], color, 1)
		
            for i in range(len(result['regions'])):
                color = [0,0,255]
                blendfact = (result['regions'][i]['confidence'] - 0.3)/0.7
                color[0] = 255*blendfact
                color[2] = 255*(1-blendfact)
                if overlaps[i]>0:
                    color[1] = 255
                color = tuple(color)
                inputImgOrig = cv2.rectangle(inputImgOrig, [int(result['regions'][i]['box']['tl_x']*inputImgOrig.shape[1]), int(result['regions'][i]['box']['tl_y']*inputImgOrig.shape[0])], [int(result['regions'][i]['box']['br_x']*inputImgOrig.shape[1]),int(result['regions'][i]['box']['br_y']*inputImgOrig.shape[0])], color, 3)
 				
            bbox = bestBbox
            print(bbox)
            if len(bbox)>0:
                inputImgOrig = cv2.rectangle(inputImgOrig, [bbox[0], bbox[1]], [bbox[0]+bbox[2],bbox[1]+bbox[3]], (0, 255, 0), 1)

            if args.showimg:
                cv2.imshow("faces", inputImgOrig )
                cv2.waitKey()
				
            if args.writeimg:
                cv2.imwrite('./faces/'+img['file_name'],inputImgOrig)

    return facedict			
   
	
def main():
    
    args = parse_args()
	
    if args.personjsonfile is not None:
	
        f = open(args.personjsonfile)
        anno = json.load(f)   
	
        facedict = processFile(anno,args)
	
        if args.outdir is not None:
            writeCSV(args.outdir+'/faces.csv',facedict)
			
    elif args.personjsonfilepattern is not None:
	
        files = glob.glob(args.personjsonfilepattern)
		
        for filename in files:
            f = open(filename)
            anno = json.load(f)   
	
            facedict = processFile(anno,args)
	
            if args.outdir is not None:
                namepart = filename.split('/')[-1]
                namepart = namepart.split('\\')[-1]
                namepart = '.'.join(namepart.split('.')[:-1])
                writeCSV(args.outdir+'/faces_'+namepart+'.csv',facedict)


if __name__ == "__main__":
    main()


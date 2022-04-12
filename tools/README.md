# People@Places Toolchain

This directory contains the toolchain to produce the annotations and additional images of the People@Places dataset. The tools may be useful to create similar annotations for other datasets.

## Dependencies

### Pytorch Image Models (TIMM)

[TIMM](https://github.com/rwightman/pytorch-image-models/tree/master/timm) is used as framework for training our classifiers for the Places365 superclasses, bustle and shot type classification.

For the using some of the features in the paper, such as the different variants of classification heads and training the classifier only, uss this [fork of TIMM](XXXXX).

### YOLOv4-CSP

For object detection, an implementation of [YOLOv4-CSP](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp) is used. We use a model trained on the MS COCO dataset. 

YOLOv4-CSP is run independently on the images. The output to be further processed by our toolchain is also expected as bounding box annotations of the detections, using the MS COCO set of classes.

### RetinaFace

For face detection, we use [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) with a model trained on WIDER Face. The detector code is called from one of the provided scripts (`run_face_places365.py`), and the root of the RetinaFace installation must be set in the script.

### ROMP

For human pose detection we use [ROMP](https://github.com/Arthur151/ROMP) with a model trained on 3D Poses in the Wild (3DPW). The detector code is called from one of the provided scripts (`run_pose_romp_places365.py`), and the root of the ROMP installation and trained model path must be set in the script.

## Steps to run the extraction pipeline

### Object detection

Run YOLOv4-CSP and store the output as MS COCO JSON files. We used a detection threshold of 0.1.

### Process object detections

Run `create_places365_annotations.py` with the following arguments:

```
--task process_detections --objjsonfile <YOLO JSON output> --pos-thresh 0.5 --img_prefix /prefix/path/in/jsons --outdir <annotation output directory>
```

`pos-thresh` denotes the threshold to accept an annotation as positive sample, any detection present with lower confidence will be considered unreliable.

Passing `--anno_discarded` will create a separate output file for discarded images (i.e. with detections below `pos_thresh`).

Images can be shown with `--show_img`, and images with bustle classifications can be shown with `--showclass`. Both require passing the path from which the images can be read using `--imgbasepath <path>`. 

A CSV file with bustle annotations and a COCO JSON file with the tallest person detections will be created as output of this step.

### Run face detector

Run `create_face_places365` with the following arguments:

```
--personjsonfile <YOLO JSON output> --imgbasepath /path/to/images --outdir <annotation output directory> 
``` 

`--personjsonfile` can be replaced with `--personjsonfilepattern` to process all files in a directory matching the pattern. The output files will be numbered consecutively.

The result images (overlaid face bounding box) can be displayed using `--showimg` or written to `./faces` using `--writeimg`.

### Run human pose detector

Run `create_pose_romp_places365` with the following arguments:

```
--personjsonfile <YOLO JSON output> --imgbasepath /path/to/images --outdir <annotation output directory> 
``` 

`--personjsonfile` can be replaced with `--personjsonfilepattern` to process all files in a directory matching the pattern. The output files will be numbered consecutively.

The result images (overlaid skeleton) can be displayed using `--showimg` or written to `./skeletons` using `--writeimg`.

### Process face and pose detections

Run `create_places365_annotations.py` with the following arguments:

```
--task process_facepose --objjsonfile <YOLO JSON output> --facefile <face detection output> --posefile <pose detection output> --outdir <annotation output directory>
```

### Create additional extreme close-up images

We found that extreme close-up images are underrepresented in Places365. Additional ones can be created by cropping from close-up images.

Run `create_places365_annotations.py` with the following arguments:

```
--task create_ecu --annotationfile <annotation CSV> --facefile <face detection output> --imgbasepath /path/to/images -outdir <annotion CSV output directory> --imgoutdir /path/to/output/cropped/images
```

### Manual verification using CVAT

We used [CVAT](https://github.com/openvinotoolkit/cvat) for manual verification of the annotation on the validation set. This is done by setting up an annotation task in CVAT with the images to be checked, and exporting the task (which produces a ZIP file containing a JSON annotation file). This JSON annotation file is replaced by a file containing the automatic annotation from the toolchain so far by running `create_places365_annotations.py` with the following arguments:

```
--task cvatimport --annotationfile <exported annotation JSON> --imagelist <exported image list> --outdir <modified annotation JSON will be written here>
```

After updating the annotion JSON file in the ZIP repository, the task can be re-imported into CVAT, and the annotations can be checked and modified. Then the task is exported again, and the annotation CSV file is produced from the exported annotation JSON file. Run `create_places365_annotations.py` with the following arguments:
 
```
--task cvat2csv --annotationfile <exported annotation JSON> --imagelist <exported image list> --outdir <modified annotation JSON will be written here>
```

### Sample dataset

In order to sample a dataset with valid annotations, run `create_places365_annotations.py` with the following arguments:

```
--task sample --annotationfile <annotation CSV file> --outdir <annotation file for sampled subset will be written here>
```

`--annotationfile` can be replaced with `--annotationfilepattern` to process all files in a directory matching the pattern. 

`--exlcude` may be used to provide an annotation CSV file of items to be excluded from sampling.

`--popcls` and `shotcls` may be used to provide a comma separated list of class IDs for bustle and shot type classes. Only items for those classes will be sampled.

If directories are provided with `--imgbasepath` and `--imgoutdir`, the sampled images will be copied from the base path to the output directory. This is in particular needed to copy original images and extreme close-up images from a directory and the corresponding `_ecu` directory.

If `--task sample` is used instead, the sampling algorithm aims to reduce the overall number of images selected when matching the target numbers per class. This mode is however much slower.

### Prepare input for TIMM

To create a directory structure with symbolic links ready to be used by TIMM, run `create_places365_annotations.py` with the following arguments:

```
--task timm --annotationfile <annotation CSV file> --imgsourcedir /path/to/images --basedir /output/dir/root --numperclass <nr of items per class> --classtype pop|shot
```

If `--binary-pop` is specified, then bustle annotations are grouped into a binary unpopulated vs. populated classification.

## Utilities

Some other utility functions are supported by the database preparation script.

### Statistics

To print the number of instances per class in the dataset, as well as the co-occurrence of bustle and shot type classes, run `create_places365_annotations.py` with the following arguments:

```
--task stats --annotationfile <annotation CSV file> 
```

### Evaluation

To evaluate accuracy of one annotation file against another one (e.g., before an after manual verification), run `create_places365_annotations.py` with the following arguments:

```
--task stats --annotationfile <annotation CSV file treated as ground truth> --annotationfile2 <annotation CSV file> 
```

If `--acc-pm1` is set, classifications into classes with class label off by 1 will be counted as correct.

## License

The tools provided here are released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
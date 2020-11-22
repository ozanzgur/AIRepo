to_import = [
    'utils',
    'object_detection.yolov3.yoloutils',
    'object_detection.yolov3.model'
    ]

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import logging
logger = logging.getLogger('pipeline')

import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

class YoloDataLoader:
    @utils.catch('YOLODATALOADER_INITERROR')
    def __init__(self, **kwargs):        
        def_args = dict(
            data_dir = '',
            anchors_name = 'regular_yolo_anchors.txt',
            classes_name = 'yolo_classes.txt',
            annotation_name = 'yolo_annot.txt',
            font_name = 'FiraMono-Medium.otf',
            data_has_offline_augmentation = False,
            batch_size = 6,
            input_shape = (832, 832),
            val_split = 0.15,
            debug = False,
            augmentation = True,
            preprocessor_method = 'transform'
        )

        self.anchors_path = os.path.join(self.data_dir, self.anchors_name)
        self.classes_path = os.path.join(self.data_dir, self.classes_name)
        self.annotation_path = os.path.join(self.data_dir, self.annotation_name)
        self.font_path = os.path.join(self.data_dir, self.font_name)
        
        # Use arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
        
        # Read annotations
        logger.info('YOLODATALOADER: Reading annotations...')
        with open(self.annotation_path) as f:
            lines = f.readlines()
        
        self.class_names = YoloDataLoader.get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = YoloDataLoader.get_anchors(self.anchors_path)
        
        # Shuffle and split data into train, val
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines)*self.val_split)
        num_train = len(lines) - num_val
        
        train_lines = []
        val_lines = []
        
        # Separate train and validation
        if self.data_has_offline_augmentation:
            logger.warning('WARNING: selecting files based on names. (data_has_offline_augmentation = True)')
            
            # Will use only augmented data
            augmented_lines = [l for l in lines if '--' in l]
            original_lines = [l for l in lines if '--' not in l]
            im_names = [line.split('--')[0] for line in augmented_lines]
            np.random.seed(12345)
            np.random.shuffle(im_names)
            np.random.seed(None)
            
            im_names = list(set(im_names))
            num_val = int(len(im_names)*self.val_split)
            num_train = len(im_names) - num_val
            train_imgs = im_names[:num_train]
            val_imgs = im_names[num_train:]
            
            train_lines = [l for l in augmented_lines if l.split('--')[0] in train_imgs]
            val_lines = [l for l in augmented_lines if l not in train_lines]
            
            train_lines = train_lines + [l for l in original_lines if '.'.join(l.split(' ')[0].split('.')[:-1]) in train_imgs]
            val_lines = val_lines + [l for l in original_lines if '.'.join(l.split(' ')[0].split('.')[:-1]) in val_imgs]
        else:
            
            # Will use all data
            train_lines = lines[:num_train]
            val_lines = lines[num_train:]
        
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.n_examples = len(train_lines) + len(val_lines)
        self.n_train = len(train_lines)
        self.n_val = len(val_lines)
        self.n_test = 0
        
        logger.info('YOLODATALOADER: Creating train generator...')
        if not self.augmentation:
            logger.warning('AUGMENTATION WAS TURNED OFF. Make sure you do offline augmentation.')
        # Create data generators for train, val
        self.train_data = YoloDataLoader.data_generator_wrapper(
            train_lines,
            self.batch_size,
            self.input_shape,
            self.anchors,
            self.num_classes,
            random = self.augmentation,
            debug = self.debug)
        
        logger.info('YOLODATALOADER: Creating val generator...')
        self.val_data = YoloDataLoader.data_generator_wrapper(
            val_lines,
            self.batch_size,
            self.input_shape,
            self.anchors,
            self.num_classes,
            random = False,
            debug = self.debug)
    
    @utils.catch('YOLODATALOADER_CALLERROR')
    def __call__(self, x, debug = False):
        return self.get_datasets(x, debug)

    @utils.catch('YOLODATALOADER_OFFLINEAUGMENTERROR')
    def offline_augment(self, rounds = 15):
        """Warning: you must first label the data and create yolo_annot.txt
        """
        logger.info('YOLODATALOADER: Reading annotations...')
        with open(self.annotation_path) as f:
            annotation_lines = f.readlines()
        
        # Select files without augmentation
        augmented_files = [al.split('--')[0] + '.' + al.split('.')[-1].split(' ')[0] for al in annotation_lines]
        augmented_files = set(augmented_files)
        annotation_lines = [al for al in annotation_lines if ('--' not in al) and (al.split(' ')[0] not in augmented_files)]
        
        logger.info(f'YOLODATALOADER: {len(annotation_lines)} files will be augmented.')
        gen = YoloDataLoader.data_generator_wrapper(
            annotation_lines,
            1,
            self.input_shape,
            self.anchors,
            self.num_classes,
            random = True,
            debug = True)
        
        new_lines = []
        
        logger.info('YOLODATALOADER: Augmenting...')
        for i_round in range(rounds):
            for line in annotation_lines:
                # Get augmented image and boxes
                image, boxes = yoloutils.get_random_data(line, self.input_shape, random=True)
                line_str_list = []
                
                # Save image
                new_image_path = line.split(' ')[0]
                extension = new_image_path.split('.')[-1]
                new_image_path = ''.join(new_image_path.split('.')[:-1]) + '--' + str(i_round) + '_large' + f'.{extension}'
                
                logger.info(f'YOLODATALOADER: Saving to {new_image_path}...')
                cv2.imwrite(new_image_path, (image*255).astype('uint8'))
                line_str_list.append(new_image_path)
                
                # Append new line
                # Get nonzero boxes
                if np.any(np.sum(boxes, axis = 1) > 1e-5):
                    boxes = boxes[np.sum(boxes, axis = 1) > 1e-5]
                    for i_box in range(len(boxes)):
                        box = boxes[i_box]
                        box_str = ','.join([str(int(box[i])) for i in range(5)])
                        line_str_list.append(box_str)
                        
                new_line = ' '.join(line_str_list) + '\n'
                new_lines.append(new_line)
                
                line_str_list = []
                # Small version
                image_small = np.array((image*255).astype('uint8'))
                h = image_small.shape[0]
                w = image_small.shape[1]
                n_dims = len(image_small.shape)

                blank = np.zeros((h*2, w*2), np.uint8) if n_dims == 2 else np.zeros((h*2, w*2, 3), np.uint8)
                blank.fill(255)
                blank[:h, :w] = image_small

                # Convert to PIL with 3 channel gray
                image_small = blank
                new_image_path = line.split(' ')[0]
                extension = new_image_path.split('.')[-1]
                new_image_path = ''.join(new_image_path.split('.')[:-1]) + '--' + str(i_round) +'_small'+ f'.{extension}'
                
                logger.info(f'YOLODATALOADER: Saving to {new_image_path}...')
                cv2.imwrite(new_image_path, image_small)
                line_str_list.append(new_image_path)
                for i_box in range(len(boxes)):
                    box = boxes[i_box]
                    box_str = ','.join([str(int(box[i])) for i in range(5)])
                    line_str_list.append(box_str)
                
                new_line = ' '.join(line_str_list) + '\n'
                new_lines.append(new_line)
                
        # Mutate yolo_annot.txt
        logger.info(f'YOLODATALOADER: Adding annotations to {self.annotation_path}...')
        with open(self.annotation_path, 'a+') as f:
            for line in new_lines:
                f.write(line)
        logger.warning(f'YOLODATALOADER: Augmentation finished. DON\'T FORGET TO RESTART PIPELINE.')
    
    @utils.catch('YOLODATALOADER_GETTRAINSAMPLEERROR')
    def get_sample(self):
        x_batch, y_batch = next(self.train_data)
        return {'x': x_batch, 'y': y_batch}
    
    @utils.catch('YOLODATALOADER_GETCLASSESERROR')
    def get_classes(classes_path):
        '''loads the classes
        '''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def get_augmented_sample(self):
        image, boxes = next(self.val_data)
        
        image = np.transpose(image, [1,0,2])
        image = self.drawboxes(Image.fromarray(image.astype('uint8')), boxes)
        
        return {'image': image, 'boxes': boxes}
    
    def drawboxes(self, image, boxes):
        font = ImageFont.truetype(font=self.font_path,
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        out_classes = boxes[0,:, 4]
        out_boxes = boxes[0,:,:4]

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[int(c)]
            box = out_boxes[i]
            score = 1.0#out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=125)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=125)
            draw.text(text_origin, label, fill=0, font=font)
            del draw
        return image
    
    @utils.catch('YOLODATALOADER_GETANCHORSERROR')
    def get_anchors(anchors_path):
        '''loads the anchors from a file
        '''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    
    @utils.catch('YOLODATALOADER_GETDATAGENERROR')
    def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random, debug = False):
        '''data generator for fit_generator
        '''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            
            # Create one batch
            for b in range(batch_size if not debug else 1):
                if i==0:
                    np.random.shuffle(annotation_lines)
                
                # Get a single preprocessed, augmented example
                if debug:
                    logger.info(annotation_lines[i])
                image, box = yoloutils.get_random_data(annotation_lines[i], input_shape, random=random)
                image_data.append(image)
                box_data.append(box)
                i = (i+1) % n
            
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            
            y_true = None
            if debug:
                y_true = box_data
            else:
                y_true = model.preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            
            # Get one batch
            if debug:
                yield image_data[0]*255, box_data
            else:
                yield [image_data, *y_true], np.zeros(batch_size)
        
            
    @utils.catch('YOLODATALOADER_GETDATAGENWRAPPERERROR')
    def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random, debug = False):
        n = len(annotation_lines)
        if n == 0 or batch_size<=0: return None
        
        return YoloDataLoader.data_generator(
            annotation_lines,
            batch_size,
            input_shape,
            anchors,
            num_classes,
            random,
            debug)

    def get_datasets(self, x, debug = False):
        if x is None:
            x = {}

        x.update({'train_data': self.train_data,
                  'val_data': self.val_data})
        return x

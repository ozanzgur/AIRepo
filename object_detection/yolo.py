to_import = [
    'utils',
    'base.model.kerasmodel',
    'object_detection.yolov3.yolomodel'
    ]

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################
import keras.backend as K
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.layers import Input, Lambda

import colorsys
from timeit import default_timer as timer
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2

import logging
logger = logging.getLogger('pipeline')

class YOLOV3(kerasmodel.BaseKerasModel):
    @utils.catch('YOLOV3_INITERROR')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Adds callbacks checkpoint, reduce_lr, early_stopping
        
        # Use arguments
        def_args = dict(
            input_shape = (416, 416, 3),
            data_dir = '',
            anchors_name = 'regular_yolo_anchors.txt',
            classes_name = 'yolo_classes.txt',
            annotation_name = 'yolo_annot.txt',
            font_name = 'FiraMono-Medium.otf',
            model_name = 'yolo_model.h5', # For test and prod
            load_pretrained = True,
            is_tiny_version = False,
            score_threshold = 0.001, # For test and prod
            iou_threshold = 0.001,  # For test and prod
            model_method = 'predict',
            steps_per_epoch = 1,
            validation_steps = 1,

            draw_preds = False,
            test_name = 'test',
            test_pred_name = 'test_pred'
            )

        # Use arguments
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
        
        # Names to paths
        self.anchors_path = os.path.join(self.data_dir, self.anchors_name)
        self.classes_path = os.path.join(self.data_dir, self.classes_name)
        self.annotation_path = os.path.join(self.data_dir, self.annotation_name)
        self.font_path = os.path.join(self.data_dir, self.font_name)
        self.model_path = os.path.join(self.data_dir, self.model_name)
        self.test_path = os.path.join(self.data_dir, self.test_name)
        self.test_pred_path = os.path.join(self.data_dir, self.test_pred_name)

        logger.info('MODEL: Creating yolov3 model.')
        
        self.gpu_num = 1
        self.sess = K.get_session()
        self.class_names = YOLOV3.get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = YOLOV3.get_anchors(self.anchors_path)
        self.model_image_size = (self.input_shape[0], self.input_shape[1])
        
        if self.model_method not in ['predict', 'test']:
            logger.info('Initializing training model.')
            if self.load_pretrained:
                logger.info('MODEL: Will freeze bottom layers.')

            if self.is_tiny_version:
                logger.info('MODEL: Creating TINY yolov3 model.')
                self.model = self.create_tiny_model(
                    self.input_shape,
                    self.anchors,
                    self.num_classes,
                    freeze_body = 2 if self.load_pretrained else 0,
                    weights_path= self.data_dir + 'tiny_yolo_weights.h5',
                    load_pretrained = self.load_pretrained)
            else:
                logger.info('MODEL: Creating REGULAR yolov3 model.')
                self.model = self.create_model(
                    self.input_shape,
                    self.anchors,
                    self.num_classes,
                    freeze_body = 2 if self.load_pretrained else 0,
                    weights_path = self.data_dir + 'yolo_weights.h5',
                    load_pretrained = self.load_pretrained)

            logger.info('MODEL: Compiling...')
            self.model.compile(optimizer=Adam(lr=3e-4), loss={
                # use custom yolo_loss Lambda layer.
                'yolo_loss': lambda y_true, y_pred: y_pred})

            logger.info('Model created.')
        else:
            logger.info(f'model_method = {self.model_method}. Will initialize prediction model.')
            self.load_test_model(self.model_path)
    
    @utils.catch('YOLOV3_CALLERROR')
    def __call__(self, x, debug = False):
        if debug:
            logger.info(f'Yolo input: {x}')
        return getattr(self, self.model_method)(x)

    @utils.catch('YOLOV3_UNFREEZEERROR')
    def unfreeeze_all_layers(self):
        """Unfreeze all layers and compile model.
        """
        # Unfreeeze
        logger.info('MODEL: Unfreeze all layers.')
        for i in range(len(self.model.layers)):
            self.model.layers[i].trainable = True
            
        # Compile model
        logger.info('MODEL: Compiling...')
        self.model.compile(optimizer = Adam(lr=1e-4),
                           loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    
    @utils.catch('YOLOV3_GETCLASSESERROR')
    def get_classes(classes_path):
        '''loads the classes
        '''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    @utils.catch('YOLOV3_GETANCHORSERROR')
    def get_anchors(anchors_path):
        '''loads the anchors from a file
        '''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    
    ### MODEL INITIALIZATION ###################################################
    
    @utils.catch('YOLOV3_INITOUTPUTERROR')
    def load_test_model(self, model_path):
        """Loads trained yolo model and creates a prediction part.
        Call this function after training, when you will test the model or predict.
        """
        model_path = os.path.expanduser(model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        logger.info(f'Loading model from: {model_path}')
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.model = load_model(model_path, compile=False)
        except:
            self.model = yolomodel.tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if self.is_tiny_version else yolomodel.yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        logger.info('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.model = multi_gpu_model(self.model, gpus=self.gpu_num)
            
        boxes, scores, classes = yolomodel.yolo_eval(
            self.model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold)
        
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
    
    @utils.catch('YOLOV3_CREATEREGULARYOLOMODELERROR')
    def create_model(self, input_shape, anchors, num_classes, load_pretrained=True,
                 freeze_body=2, weights_path = 'mlpipeline/yolo/yolov3/model_data/yolo_weights.h5'):
        '''create the training model
        '''
        K.clear_session() # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w, c = input_shape
        num_anchors = len(anchors)

        y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
            num_anchors//3, num_classes+5)) for l in range(3)]

        model_body = yolomodel.yolo_body(image_input, num_anchors//3, num_classes)
        logger.info('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            logger.info('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers)-3)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                logger.info('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolomodel.yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

        return model
    
    @utils.catch('YOLOV3_CREATETINYYOLOMODELERROR')
    def create_tiny_model(self, input_shape, anchors, num_classes, load_pretrained=True,
                          freeze_body=2, weights_path = 'mlpipeline/yolo/yolov3/model_data/tiny_yolo_weights.h5'):
        '''create the training model, for Tiny YOLOv3
        '''
        K.clear_session() # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w, c = input_shape
        num_anchors = len(anchors)

        y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
            num_anchors//2, num_classes+5)) for l in range(2)]

        model_body = yolomodel.tiny_yolo_body(image_input, num_anchors//2, num_classes)
        logger.info('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            logger.info('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze the darknet body or freeze all but 2 output layers.
                num = (20, len(model_body.layers)-2)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                logger.info('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolomodel.yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
            [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

        return model
    
    ### TRAINING ###########################################################
    
    @utils.catch('YOLOV3_FITERROR')
    def fit(self, x, **fit_params):
        
        train_data = x['train_data']
        val_data = x['val_data']

        metrics = None
        if self.load_pretrained:
            self.fit_generator(
                train_data = train_data,
                val_data = val_data,
                steps_per_epoch = self.steps_per_epoch,
                validation_steps = self.validation_steps,
                epochs = 50,
                initial_epoch = 0)
        
            # Unfreeze
            self.unfreeeze_all_layers()

            # Train all layers
            metrics = self.fit_generator(
                train_data = train_data,
                val_data = val_data,
                steps_per_epoch = self.steps_per_epoch,
                validation_steps = self.validation_steps,
                epochs=100,
                initial_epoch=50)
        else:
            # Train all layers
            metrics = self.fit_generator(
                train_data = train_data,
                val_data = val_data,
                steps_per_epoch = self.steps_per_epoch,
                validation_steps = self.validation_steps,
                epochs = 100,
                initial_epoch = 0)
        return metrics
    
    ### PREDICTION #############################################################
    @utils.catch('YOLOV3_TESTSAVEDIRERROR')
    def pred_dir(self, x, debug = False):
        """Saves predictions to directory.
        """
        # Create prediction directory
        logger.info(f'Create directory: {self.test_pred_path}')
        os.makedirs(self.test_pred_path, exist_ok=True)
        
        # Predict on all images in directory
        images = os.listdir(self.test_path)
        logger.info(f'Predict on images in {self.test_path}, n_images = {len(images)}')
        for im_name in images:
            im_path = os.path.join(self.test_path, im_name)
            im_save_path = os.path.join(self.test_pred_path, im_name)
            
            image = Image.open(im_path)
            #image = image.convert('L').convert('RGB')
            
            logger.info(f'Path: {im_path}')
            
            pred_dict = self.predict({'output': image}, debug = True)
            image_pred = pred_dict[0]['image']
            image_pred = np.array(image_pred)
            #image_pred = image_pred[:, :, ::-1]
            
            logger.info(f'Save to: {im_save_path}')
            
            # Save prediction as tif
            cv2.imwrite(im_save_path[:-4] + '.tif', image_pred)
    
    @utils.catch('YOLOV3_PREDICTERROR')
    def predict(self, x, debug = False):
        for x_example in x:
            x_example.update(self.predict_single(x_example, debug))
        return x

    @utils.catch('YOLOV3_CLOSESESSIONERROR')
    def close_session(self):
        self.sess.close()
    
    @utils.catch('YOLOV3_PREDICTSINGLEERROR')
    def predict_single(self, x, debug = False):
        image_processed = x['image']
        image_orig = x['image_orig']

        start = timer()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: image_processed,
                self.input_image_shape: [image_orig.size[1], image_orig.size[0]]
                #K.learning_phase(): 0
            })
        
        pred_t = timer()
        logger.info(f'Prediction time: {pred_t - start}')
                    
        logger.info('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        if not self.draw_preds:
            logger.info('YOLOV3: Return only predictions.')
            return {'output': (out_boxes, out_scores, out_classes)}
            
        else:
            logger.info('Return image with detection boxes drawn.')
            
            font = ImageFont.truetype(font=self.font_path,
                        size=np.floor(3e-2 * image_orig.size[1] + 0.5).astype('int32'))
            thickness = (image_orig.size[0] + image_orig.size[1]) // 300
            
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image_orig)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image_orig.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image_orig.size[0], np.floor(right + 0.5).astype('int32'))
                logger.info(f'{label}, ({left}, {top}), ({right}, {bottom})')
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                
                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=128)#self.colors[c][0])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=128)#self.colors[c][0])
                draw.text(text_origin, label, fill=0, font=font)
                del draw

            end = timer()
            logger.info(end - start)
            
            return {'pred_image': image_orig,
                    'output':(out_boxes, out_scores, out_classes)}
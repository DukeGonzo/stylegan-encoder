import os
import bz2
import PIL.Image
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K
import keras.layers as layers

import traceback

def load_images(images_list, image_size=256):
    loaded_images = list()
    for img_path in images_list:
      img = PIL.Image.open(img_path).convert('RGB').resize((image_size,image_size),PIL.Image.LANCZOS)
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

def tf_custom_l1_loss(img1,img2):
  return tf.math.reduce_mean(tf.math.abs(img2-img1), axis=None)

def tf_custom_l2_loss(img1,img2):
  return tf.sqrt(tf.math.reduce_mean(tf.math.square(img2-img1), axis=None))

def tf_custom_logcosh_loss(img1,img2):
  return tf.math.reduce_mean(tf.keras.losses.logcosh(img1,img2))

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class PerceptualModel:
    def __init__(self, args, batch_size=1, perc_model=None, facenet_model=None, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.lr = args.lr
        self.decay_rate = args.decay_rate
        self.decay_steps = args.decay_steps
        self.fastai_schedule = args.fastai
        self.iterations = args.iterations
        self.img_size = args.image_size
        self.layer = args.use_vgg_layer
        self.vgg_loss = args.use_vgg_loss
        self.face_mask = args.face_mask
        self.use_grabcut = args.use_grabcut
        self.scale_mask = args.scale_mask
        self.facenet_loss = args.use_facenet_loss
        self.mask_dir = args.mask_dir
        self.facenet_model = None
        if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
            self.vgg_loss = None
        self.pixel_loss = args.use_pixel_loss
        if (self.pixel_loss <= self.epsilon):
            self.pixel_loss = None
        self.mssim_loss = args.use_mssim_loss
        if (self.mssim_loss <= self.epsilon):
            self.mssim_loss = None
        self.lpips_loss = args.use_lpips_loss
        if (self.lpips_loss <= self.epsilon):
            self.lpips_loss = None
        self.l1_penalty = args.use_l1_penalty
        if (self.l1_penalty <= self.epsilon):
            self.l1_penalty = None
        self.batch_size = batch_size
        if perc_model is not None and self.lpips_loss is not None:
            self.perc_model = perc_model
        else:
            self.perc_model = None

        if self.facenet_loss <= self.epsilon:
            self.facenet_loss = None
        if facenet_model is not None:
            self.facenet_model = facenet_model


        self.ref_img = None
        self.ref_weight = None
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

        if self.facenet_model is not None or self.face_mask:
            import dlib
            self.detector = dlib.get_frontal_face_detector()

        if self.face_mask:
            LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                    LANDMARKS_MODEL_URL, cache_subdir='temp'))
            self.predictor = dlib.shape_predictor(landmarks_model_path)

    def compare_images(self,img1,img2):
        if self.perc_model is not None:
            return self.perc_model.get_output_for(tf.transpose(img1, perm=[0,3,2,1]), tf.transpose(img2, perm=[0,3,2,1]))
        return 0

    def add_placeholder(self, var_name):
        var_val = getattr(self, var_name)
        setattr(self, var_name + "_placeholder", tf.placeholder(var_val.dtype, shape=var_val.get_shape()))
        setattr(self, var_name + "_op", var_val.assign(getattr(self, var_name + "_placeholder")))

    def assign_placeholder(self, var_name, var_val):
        self.sess.run(getattr(self, var_name + "_op"), {getattr(self, var_name + "_placeholder"): var_val})

    def build_perceptual_model(self, generator):
        # Learning rate
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        incremented_global_step = tf.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        if self.fastai_schedule:
            self.learning_rate = tf.minimum(self.lr * 2 * tf.cast(incremented_global_step, tf.float32) / self.iterations, 
                                            self.lr * 2 * (1 - tf.cast(incremented_global_step, tf.float32) / self.iterations))
        else:
            self.learning_rate = tf.train.exponential_decay(self.lr, incremented_global_step,
                    self.decay_steps, self.decay_rate, staircase=True)
        self.sess.run([self._reset_global_step])

        generated_image_tensor = generator.generated_image
        generated_image = tf.image.resize_nearest_neighbor(generated_image_tensor,
                                                                  (self.img_size, self.img_size), align_corners=True)

        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.ref_weight = tf.get_variable('ref_weight', shape=generated_image.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder("ref_img")
        self.add_placeholder("ref_weight")

        if (self.vgg_loss is not None):
            vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
            # vgg16.summary()

            for i, l in enumerate(vgg16.layers):
                print(f'{i}, {l.output}')

            print(f'======================{self.layer}===========================')

            out = vgg16.layers[self.layer].output
            out = layers.Lambda(lambda x: (x - K.mean(x, axis=(1,2, 3), keepdims=True)) / K.std(x, axis=(1,2, 3), keepdims=True))(out)
            # out = [la.output, (-1,) for la in vgg16.layers[self.layer - 2 : self.layer+2]]
            self.perceptual_model = Model(vgg16.input, out)
            generated_img_features = self.perceptual_model(preprocess_input(self.ref_weight * generated_image))
            self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
            self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
            self.sess.run([self.features_weight.initializer, self.features_weight.initializer])
            self.add_placeholder("ref_img_features")
            self.add_placeholder("features_weight")

        self.loss = 0
        # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            self.loss += self.vgg_loss * tf_custom_l1_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
        # + logcosh loss on image pixels
        # if (self.pixel_loss is not None):
        #     self.loss += self.pixel_loss * tf_custom_logcosh_loss(self.ref_weight * self.ref_img, self.ref_weight * generated_image)
        # + sobelloss
        if (self.pixel_loss is not None):
            # generated_image_gray =  tf.image.rgb_to_grayscale(generated_image)
            generated_image_gray =  generated_image
            gen_sob =  tf.image.sobel_edges(generated_image_gray)
            # ref_img =  tf.image.rgb_to_grayscale(self.ref_img)
            ref_img =  self.ref_img
            ref_sob =  tf.image.sobel_edges(ref_img)

            self.loss += self.pixel_loss * tf_custom_l1_loss(ref_sob, gen_sob)
            # self.loss += self.pixel_loss * tf_custom_logcosh_loss(ref_sob, gen_sob)
        # + MS-SIM loss on image pixels
        if (self.mssim_loss is not None):
            self.loss += self.mssim_loss * tf.math.reduce_mean(1-tf.image.ssim_multiscale(self.ref_weight * self.ref_img, self.ref_weight * generated_image, 1))
        # + extra perceptual loss on image pixels
        if self.perc_model is not None and self.lpips_loss is not None:
            self.loss += self.lpips_loss * tf.math.reduce_mean(self.compare_images(self.ref_weight * self.ref_img, self.ref_weight * generated_image))

        if self.facenet_model is not None and self.facenet_loss is not None:
            self.ref_boxes = tf.get_variable('ref_boxes', (self.batch_size, 4), dtype='float32')
            self.add_placeholder("ref_boxes")

            #crop 160:
            # [82: 192, 72: 182]
            # croped_ref = tf.image.crop_to_bounding_box(self.ref_img, 82, 72, 110, 110)
            # res_ref = tf.image.resize_images(croped_ref, (160,160))
            batch_size = tf.shape(self.ref_img)[0]
            res_ref = tf.image.crop_and_resize(self.ref_img, self.ref_boxes, tf.range(0, batch_size, dtype='int32'), (160,160))

            # croped_generated_image = tf.image.crop_to_bounding_box(generated_image, 82, 72, 110, 110)
            # res_generated_image= tf.image.resize_images(croped_generated_image, (160,160))
            res_generated_image = tf.image.crop_and_resize(generated_image, self.ref_boxes, tf.range(0, batch_size, dtype='int32'), (160,160))

            # self.facenet_model.summary()
            #loss
            facenet_ref = self.facenet_model(self.prewhiten(res_ref))
            # facenet_ref_flip = self.facenet_model(self.flip(self.prewhiten(res_ref)))

            # facenet_ref += facenet_ref_flip
            # facenet_ref /= 2.0

            facenet_gen = self.facenet_model(self.prewhiten(res_generated_image))
            # facenet_gen_flip = self.facenet_model(self.flip(self.prewhiten(res_generated_image)))

            # facenet_gen += facenet_gen_flip
            # facenet_gen /= 2.0

            # facenet_ref=tf.math.l2_normalize(facenet_ref, axis=-1)
            # facenet_gen=tf.math.l2_normalize(facenet_gen, axis=-1)

            # self.loss += self.facenet_loss * (tf_custom_l1_loss(facenet_ref, facenet_gen) + tf_custom_l1_loss(facenet_ref_flip, facenet_gen_flip)) / 2.0
            self.loss += self.facenet_loss * tf_custom_l1_loss(facenet_ref, facenet_gen)

        # + L1 penalty on dlatent weights

        if self.l1_penalty is not None:
            dlatent_avg = generator.dlatent_variable-generator.get_dlatent_avg()
            print(dlatent_avg)
            self.loss += self.l1_penalty * 512 * tf.sqrt(tf.math.reduce_mean(tf.math.square(dlatent_avg)))

    def prewhiten(self, x):
        mean = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        std = K.std(x, axis=(1, 2, 3), keepdims=True)
        std_adj = tf.maximum(std, 1.0 / np.sqrt(512))
        return (x - mean) / std_adj

    def flip(self, x):
        return tf.image.flip_left_right(x)


    def generate_face_mask(self, im):
        from imutils import face_utils
        import cv2
        rects = self.detector(im, 1)
        # loop over the face detections
        for (j, rect) in enumerate(rects):
            """
            Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            """
            shape = self.predictor(im, rect)
            shape = face_utils.shape_to_np(shape)

            # we extract the face
            vertices = cv2.convexHull(shape)
            mask = np.zeros(im.shape[:2],np.uint8)
            cv2.fillConvexPoly(mask, vertices, 1)
            if self.use_grabcut:
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (0,0,im.shape[1],im.shape[2])
                (x,y),radius = cv2.minEnclosingCircle(vertices)
                center = (int(x),int(y))
                radius = int(radius*self.scale_mask)
                mask = cv2.circle(mask,center,radius,cv2.GC_PR_FGD,-1)
                cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
                cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
                mask = np.where((mask==2)|(mask==0),0,1)
            return mask

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = None
        if self.perceptual_model is not None:
            image_features = self.perceptual_model.predict_on_batch(preprocess_input(loaded_image))
            weight_mask = np.ones(self.features_weight.shape)

        if self.face_mask:
            image_mask = np.zeros(self.ref_weight.shape)
            for (i, im) in enumerate(loaded_image):
                try:
                    _, img_name = os.path.split(images_list[i])
                    mask_img = os.path.join(self.mask_dir, f'{img_name}')
                    if (os.path.isfile(mask_img)):
                        print("Loading mask " + mask_img)
                        imask = PIL.Image.open(mask_img).convert('L')
                        mask = np.array(imask)/255
                        mask = np.expand_dims(mask,axis=-1)
                    else:
                        mask = self.generate_face_mask(im)
                        imask = (255*mask).astype('uint8')
                        imask = PIL.Image.fromarray(imask, 'L')
                        print("Saving mask " + mask_img)
                        imask.save(mask_img, 'PNG')
                        mask = np.expand_dims(mask,axis=-1)
                    mask = np.ones(im.shape,np.float32) * mask
                except Exception as e:
                    print("Exception in mask handling for " + mask_img)
                    traceback.print_exc()
                    mask = np.ones(im.shape[:2],np.uint8)
                    mask = np.ones(im.shape,np.float32) * np.expand_dims(mask,axis=-1)
                image_mask[i] = mask
            img = None
        else:
            image_mask = np.ones(self.ref_weight.shape)

        boxes = None
        if self.facenet_model:
            boxes = np.zeros((len(images_list), 4))
            w = loaded_image.shape[1]
            h = loaded_image.shape[2]
            for i, img in enumerate(loaded_image):
                faces = self.detector(img)
                if len(faces) == 0:
                    continue
                x1 = faces[0].tl_corner().x
                y1 = faces[0].tl_corner().y
                x2 = faces[0].br_corner().x
                y2 = faces[0].br_corner().y

                boxes[i] = [y1/h, x1/w, y2/h, x2/w]

        if len(images_list) != self.batch_size:
            if image_features is not None:
                features_space = list(self.features_weight.shape[1:])
                existing_features_shape = [len(images_list)] + features_space
                empty_features_shape = [self.batch_size - len(images_list)] + features_space
                existing_examples = np.ones(shape=existing_features_shape)
                empty_examples = np.zeros(shape=empty_features_shape)
                weight_mask = np.vstack([existing_examples, empty_examples])
                image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

            images_space = list(self.ref_weight.shape[1:])
            existing_images_space = [len(images_list)] + images_space
            empty_images_space = [self.batch_size - len(images_list)] + images_space
            existing_images = np.ones(shape=existing_images_space)
            empty_images = np.zeros(shape=empty_images_space)
            image_mask = image_mask * np.vstack([existing_images, empty_images])
            loaded_image = np.vstack([loaded_image, np.zeros(empty_images_space)])
            if boxes is not None:
                boxes = np.vstack([boxes, np.zeros([self.batch_size - len(images_list)])])                

        if image_features is not None:
            self.assign_placeholder("features_weight", weight_mask)
            self.assign_placeholder("ref_img_features", image_features)
        self.assign_placeholder("ref_weight", image_mask)
        self.assign_placeholder("ref_img", loaded_image)
        if boxes is not None:
            self.assign_placeholder("ref_boxes", boxes)

    def optimize(self, vars_to_optimize, iterations=200):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        # optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        self.sess.run(self._reset_global_step)
        fetch_ops = [min_op, self.loss, self.learning_rate]
        for _ in range(iterations):
            _, loss, lr = self.sess.run(fetch_ops)
            yield {"loss":loss, "lr": lr}

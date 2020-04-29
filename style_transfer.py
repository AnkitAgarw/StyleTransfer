def style_transfer(loc):
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    print(frame.shape)
    capture.release()
    
    
    import tensorflow as tf
    
    sess=tf.Session()
    g = tf.Graph()
    #First let's load meta graph and restore weights
    
    WEIGHTS_INIT_STDEV = .1
    def _residual_block(net, filter_size=3):
        tmp = _conv_layer(net, 128, filter_size, 1)
        return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)
    
    def net(image):
        conv1 = _conv_layer(image, 32, 9, 1)
        conv2 = _conv_layer(conv1, 64, 3, 2)
        conv3 = _conv_layer(conv2, 128, 3, 2)
        resid1 = _residual_block(conv3, 3)
        resid2 = _residual_block(resid1, 3)
        resid3 = _residual_block(resid2, 3)
        resid4 = _residual_block(resid3, 3)
        resid5 = _residual_block(resid4, 3)
        conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
        conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
        conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
        preds = tf.nn.tanh(conv_t3) * 150 + 255./2
        return preds
    
    def _conv_layer(net, num_filters, filter_size, strides, relu=True):
        weights_init = _conv_init_vars(net, num_filters, filter_size)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
        net = _instance_norm(net)
        if relu:
            net = tf.nn.relu(net)
    
        return net
    
    def _conv_tranpose_layer(net, num_filters, filter_size, strides):
        weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    
        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])
    
        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]
    
        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = _instance_norm(net)
        return tf.nn.relu(net)
    
    
    def _instance_norm(net, train=True):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift = tf.Variable(tf.zeros(var_shape))
        scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        return scale * normalized + shift
    
    def _conv_init_vars(net, out_channels, filter_size, transpose=False):
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]
    
        weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
        return weights_init
    
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(0)
    
    ret, input_img= capture.read()
    img_placeholder = tf.placeholder(tf.float32, shape=(1,)+input_img.shape,
                                      name='img_placeholder')
    preds = net(img_placeholder)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(loc))
    
    while(True):
          
        ret, X= capture.read()
        X=cv2. cvtColor(X, cv2. COLOR_BGR2RGB)
        _preds = sess.run(preds, feed_dict={img_placeholder:X.reshape((1,)+input_img.shape)})
        _preds = np.clip(_preds, 0, 255)
        _preds = _preds.astype(np.uint8)
        cv2.namedWindow("a", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        cv2.imshow('a', _preds.reshape((_preds.shape[1],_preds.shape[2],_preds.shape[3])))
    
          
        if cv2.waitKey(1) == 27:
            break
      
    capture.release()
    cv2.destroyAllWindows()


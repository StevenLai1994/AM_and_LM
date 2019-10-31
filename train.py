import os
import tensorflow as tf
from utils import get_data, data_hparams
from keras.callbacks import ModelCheckpoint


# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'train'
# data_args.data_path = '../dataset/'
data_args.thchs30 = True
data_args.aishell = True
data_args.prime = True
data_args.stcmd = False
data_args.batch_size = 8
data_args.data_length = 16
data_args.shuffle = True
train_data = get_data(data_args)

# 0.准备验证所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'dev'
# data_args.data_path = '../dataset/'
data_args.thchs30 = True
data_args.aishell = True
data_args.prime = False
data_args.stcmd = False
data_args.batch_size = 8
data_args.data_length = 16
data_args.shuffle = True
dev_data = get_data(data_args)


def train_am(epochs):
    # 1.声学模型训练-----------------------------------
    from model_speech.cnn_ctc import Am, am_hparams
    am_args = am_hparams()
    am_args.vocab_size = len(train_data.am_vocab)
    am_args.gpu_nums = 1
    am_args.lr = 0.0008
    am_args.is_training = True
    am = Am(am_args)

    if os.path.exists('logs_am') and not os.listdir('logs_am'):
        model = os.listdir('logs_am')[0]
        am.ctc_model.load_weights(os.path.join('logs_am', model))

    batch_num = len(train_data.wav_lst) // train_data.batch_size

    # checkpoint
    ckpt = "model_{val_loss:.3f}_{epoch:04d}.h5"
    checkpoint = ModelCheckpoint(os.path.join('logs_am', ckpt), monitor='val_loss', save_weights_only=True, verbose=1, save_best_only=True)

    batch = train_data.get_am_batch()
    dev_batch = dev_data.get_am_batch()

    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=2)
    am.ctc_model.save_weights('logs_am/model.h5')


def train_lm(epochs):
    # 2.语言模型训练-------------------------------------------
    from model_language.transformer import Lm, lm_hparams
    lm_args = lm_hparams()
    lm_args.num_heads = 8
    lm_args.num_blocks = 6
    lm_args.input_vocab_size = len(train_data.pny_vocab)
    lm_args.label_vocab_size = len(train_data.han_vocab)
    lm_args.max_length = 100
    lm_args.hidden_units = 512
    lm_args.dropout_rate = 0.2
    lm_args.lr = 0.0003
    lm_args.is_training = True
    lm = Lm(lm_args)

    batch_num = len(train_data.wav_lst) // train_data.batch_size
    dev_batch_num = len(dev_data.wav_lst) // dev_data.batch_size

    with lm.graph.as_default():
        saver =tf.train.Saver(max_to_keep=5)
    with tf.Session(graph=lm.graph) as sess:
        add_num = -1
        if os.path.exists('logs_lm/checkpoint'):
            # 恢复变量
            import re
            model_file = tf.train.latest_checkpoint('logs_lm')
            add_num = int(re.findall('\d+', model_file)[0])
            print("================================restore latest save model, epoch=%d==============================" % add_num)
            saver.restore(sess, model_file)
        else:
            # 初始化变量
            sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        min_loss = 100
        writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())

        for k in range(epochs - add_num - 1):
            #训练==============================================
            total_loss = 0
            batch = train_data.get_lm_batch()                                                                                                                                                 
            for i in range(batch_num):
                input_batch, label_batch = next(batch)
                feed = {lm.x: input_batch, lm.y: label_batch}                                                                                                                                 
                cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)                                                                                                                 
                total_loss += cost                                                                                                                                                            
                if (k * batch_num + i) % 10 == 0:
                    rs=sess.run(merged, feed_dict=feed)                                                                                                                                       
                    writer.add_summary(rs, k * batch_num + i)                                                                                                                                 
                if (i+1) % 500 == 0:
                    print('epoch:{:04d}, step:{:05d}, train_loss:{:0.3f}'.format(k, i, total_loss/i))
            epoch_loss = total_loss/batch_num                                                                                                                                                 
            print('epochs', k+1, ': average loss = ', epoch_loss) 

            #验证==============================================                                                                                                                                                             
            dev_batch = dev_data.get_lm_batch()                                                                                                                                               
            for i in range(dev_batch_num):
                total_loss = 0
                input_batch, label_batch = next(dev_batch)
                feed = {lm.x: input_batch, lm.y: label_batch}                                                                                                                                 
                cost, _ = sess.run([lm.mean_loss, lm.train_op], feed_dict=feed)                                                                                                               
                total_loss += cost                                                                                                                                                            
                if (i+1) % 500 == 0:
                    print("dev_step:{:05d}, dev_loss:{:0.3f}".format(i, total_loss/i))
            print('dev_loss:{:.3f}'.format(total_loss / dev_batch_num))
            if total_loss/batch_num < min_loss:
                min_loss = total_loss/batch_num
                print("save model")                                                                                                                                    
                saver.save(sess, 'logs_lm/model_epoch_{:04d}_val_loss_{:0.3f}'.format(k + add_num + 1, min_loss))

        writer.close()                                                 

if __name__ == '__main__':
    train_lm(10)
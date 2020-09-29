from matplotlib import pyplot




def See_loss_curve(log_file, loss_name, num):
    loss_list = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            if loss_name in line:
                index = line.find(loss_name) + num
                loss_value = float(line[index:index+6])
                loss_list.append(loss_value)
                # print(loss_value)
        pyplot.plot(loss_list)
        pyplot.ylim(0,4)
        print(loss_list)
        print(len(loss_list))
        pyplot.show()



if __name__ == '__main__':
    log_file = r'D:\Low level for Real\cycle GAN\pytorch-CycleGAN-and-pix2pix\checkpoints\maps_cyclegan\loss_log.txt'
    See_loss_curve(log_file,'cycle_A', 8)

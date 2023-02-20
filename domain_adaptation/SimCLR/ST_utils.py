import torch
import constants
import numpy as np
import domain_adaptation.SimCLR.transformmasks as transformmasks

ignore_label = constants.IGNORE_INDEX



def update_ema_variables(ema_backbone, ema_head, backbone, head, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    
    for ema_param, param in zip(ema_backbone.parameters(), backbone.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    for ema_param, param in zip(ema_head.parameters(), head.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_backbone, ema_head





def define_mix_type(inputs_u_w, max_probs, argmax_u_w, mix_mask):
    if mix_mask == "class":

        for image_i in range(len(argmax_u_w)):
            classes = torch.unique(argmax_u_w[image_i])
            classes = classes[classes != ignore_label]
            nclasses = classes.shape[0]
            classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
            if image_i == 0:
                MixMask = transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()
            else:
                MixMask = torch.cat((MixMask, transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()))

    elif mix_mask == 'cut':
        img_size = inputs_u_w.shape[2:4]
        for image_i in range(len(argmax_u_w)):
            if image_i == 0:
                MixMask = torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).cuda().float()
            else:
                MixMask = torch.cat((MixMask, torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).cuda().float()))

    elif mix_mask == "cow":
        img_size = inputs_u_w.shape[2:4]
        sigma_min = 8
        sigma_max = 32
        p_min = 0.5
        p_max = 0.5
        for image_i in range(len(argmax_u_w)):
            sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))     # Random sigma
            p = np.random.uniform(p_min, p_max)     # Random p
            if image_i == 0:
                MixMask = torch.from_numpy(transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(0).cuda().float()
            else:
                MixMask = torch.cat((MixMask,torch.from_numpy(transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(0).cuda().float()))

    elif mix_mask == None:
        MixMask = torch.ones((inputs_u_w.shape))


    return MixMask

def sigmoid_ramp_up(iter, max_iter):
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - iter / max_iter) ** 2)


def define_pixel_weight( max_probs, pseudo_label, pixel_weight = None, i_iter = 0):
    if pixel_weight == "threshold_uniform":
        unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
    
    elif pixel_weight == "threshold":
        pixelWiseWeight = max_probs.ge(0.968).long().cuda()
    
    elif pixel_weight == 'sigmoid':
        max_iter = 10000
        pixelWiseWeight = sigmoid_ramp_up(i_iter, max_iter) * torch.ones(max_probs.shape).cuda()
    
    elif pixel_weight == None:
        pixelWiseWeight = torch.ones(max_probs.shape).cuda()

    return pixelWiseWeight
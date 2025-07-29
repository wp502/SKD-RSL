import math
import numpy as np
import torch

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    return np.dot(np.dot(H, K), H)
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)



# GPU Version
def centering_GPU(K):
    n = K.shape[0]
    unit = torch.ones([n, n]).cuda()
    I = torch.eye(n).cuda()
    H = I - unit / n

    # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    return torch.mm(torch.mm(H, K), H)
    # return np.dot(H, K)  # KH

def linear_HSIC_GPU(X, Y):
    L_X = torch.mm(X, X.t())
    L_Y = torch.mm(Y, Y.t())
    return torch.sum(centering_GPU(L_X) * centering_GPU(L_Y))

def linear_CKA_GPU(X, Y):
    hsic = linear_HSIC_GPU(X, Y)
    var1 = torch.sqrt(linear_HSIC_GPU(X, X))
    var2 = torch.sqrt(linear_HSIC_GPU(Y, Y))

    return hsic / (var1 * var2)



if __name__ == '__main__':
    '''
    如果用GPU运算出来的结果是负值，则可以切换为CPU运算。
    GPU_linear: linear_CKA_GPU
    CPU_linear: linear_CKA
    '''
    # X = np.random.randn(100, 64)
    # Y = np.random.randn(100, 64)

    # X_tensor = torch.randn(128, 65536)
    # Y_tensor = torch.randn(128, 65536)
    # X_tensor = torch.randn(128, 50176)
    # Y_tensor = torch.randn(128, 18816)
    #
    # X_ndarray = X_tensor.numpy()
    # Y_ndarray = Y_tensor.numpy()

    X_ndarray = np.load('./save/feat/feat_fdbk/t_fdbk_4.npy')
    Y_ndarray = np.load('./save/feat/feat_fdbk/s_fdbk_4.npy')
    X_tensor = torch.Tensor(X_ndarray)
    Y_tensor = torch.Tensor(Y_ndarray)
    print(X_tensor.shape)
    print(Y_tensor.shape)


    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X_ndarray, Y_ndarray)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X_ndarray, X_ndarray)))

    # print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X_ndarray, Y_ndarray)))
    # print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X_ndarray, X_ndarray)))

    print('Linear CKA GPU, between X and Y: {}'.format(linear_CKA_GPU(X_tensor.cuda(), Y_tensor.cuda())))
    print('Linear CKA GPU, between X and X: {}'.format(linear_CKA_GPU(X_tensor.cuda(), X_tensor.cuda())))



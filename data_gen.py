import numpy as np
from ir_match_matrix import get_one2many_mm, get_one2one_mm


def fft_ind_gen(n):
    a = list(range(0, int(n / 2 + 1)))
    b = list(range(1, int(n / 2)))
    b.reverse()
    b = [-i for i in b]
    return a + b


def gaussian_random_field(pk=lambda k: k ** -3.0, size1=100, size2=100, anisotropy=True):
    def pk2(kx_, ky_):
        if kx_ == 0 and ky_ == 0:
            return 0.0
        if anisotropy:
            if kx_ != 0 and ky_ != 0:
                return 0.0
        return np.sqrt(pk(np.sqrt(kx_ ** 2 + ky_ ** 2)))

    noise = np.fft.fft2(np.random.normal(size=(size1, size2)))
    amplitude = np.zeros((size1, size2))
    for i, kx in enumerate(fft_ind_gen(size1)):
        for j, ky in enumerate(fft_ind_gen(size2)):
            amplitude[i, j] = pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)


def next_batch_gaussian(bs, h, w, anisotropy=True):
    x = []
    for i in range(bs):
        o = gaussian_random_field(pk=lambda k: k ** -4.0, size1=h, size2=w, anisotropy=anisotropy).real
        x.append(o)
    x = np.array(x)
    y = np.roll(x, shift=-1, axis=2)
    y[:, :, -1] = 0.0
    return x, y, np.ones_like(x)


def next_batch_ir(bs, h, w, **kwargs):
    batch = get_one2many_mm(bs, h, w, **kwargs)
    #for mm in batch[0]:
    #    print(mm)
    #    input()
    return batch


def next_batch(data_type, *args, **kwargs):
    if data_type == 'gau':
        return next_batch_gaussian(*args, **kwargs)
    elif data_type == 'ir':
        return next_batch_ir(*args, **kwargs)
    else:
        raise Exception('not supported data type (should be one of "gau", "ir").')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for alpha in [-4.0, -2.0, -0.0]:
        out = gaussian_random_field(pk=lambda k: k ** alpha, size1=256, size2=256, anisotropy=False)
        plt.figure()
        plt.imshow(out.real, cmap='jet', interpolation='none')
    plt.show()

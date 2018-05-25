# encoding: utf-8
"""
 Author: govind
 Date: 2018/5/25
 Description:

    1. Mapping data to higher dimensions with kernels.
      mapping from one feature space to another feature space.Usually, this mapping goes
       from a lowerdimensional feature space to a higher-dimensional space.
    2.Inner products are two vectors multiplied together to yield a scalar or single number.
      Replacing the inner product with a kernel is known as the kernel trick or kernel substation

    3.Radial Bias Function
                K(x, y) = exp(- ||x-y||** 2 / (2 * sigma ** 2))

    This Gaussian version maps the data from its feature space to a higher feature space, infinite
    dimensional to be specific.
    因为存在有些数据不能通过上述公式求得两样本点的距离，因此需要对使用哪种方式进行核转换进行分类处理。
"""

def kernel_transform(X, A, k_tuple):
    m, n = X.shape

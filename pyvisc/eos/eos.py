#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import pyopencl as cl
import numpy as np
from scipy.interpolate import interp1d

class Eos(object):
    '''create eos table for hydrodynamic simulation;
    the (ed, pr, T, s) is stored in image2d buffer
    for fast linear interpolation'''
    def __init__(self, cfg):
        # information of the eos table
        if cfg.IEOS == 2:
            import wb
            self.ed = wb.ed
            self.pr = wb.pr
            self.T = wb.T
            self.ed_start = wb.ed_start
            self.ed_step = wb.ed_step
            self.num_of_ed = wb.num_ed
        else:
        #else cfg.IEOS == 3:
            import glueball
            self.ed = glueball.ed
            self.pr = glueball.pr
            self.T = glueball.T
            self.ed_start = glueball.ed_start
            self.ed_step = glueball.ed_step
            self.num_of_ed = glueball.num_ed

        self.cfg = cfg
        self.s = (self.ed + self.pr)/self.T

        # interpolation functions
        self.f_ed = interp1d(self.T, self.ed)
        self.f_T = interp1d(self.ed, self.T)
        self.f_P = interp1d(self.ed, self.pr)

    def create_table(self, ctx, compile_options, nrow=200, ncol=1000):
        '''store the eos (ed, pr, T, s) in image2d_t table for fast
        linear interpolation,
        add some information to compile_options for EOS table'''
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
        src = np.array(zip(self.ed, self.pr, self.T, self.s),
                 dtype=np.float32).reshape(nrow, ncol, 4)

        eos_table = cl.image_from_array(ctx, src, 4)
        compile_options.append('-D EOS_ED_START={value}f'.format(
                                                 value=self.ed_start))
        compile_options.append('-D EOS_ED_STEP={value}f'.format(
                                                 value=self.ed_step))
        compile_options.append('-D EOS_NUM_ED={value}'.format(
                                                 value=self.num_of_ed))
        compile_options.append('-D EOS_NUM_OF_ROWS=%s'%nrow)
        compile_options.append('-D EOS_NUM_OF_COLS=%s'%ncol)
        self.compile_options = compile_options
        return eos_table

    def test_eos(self, test_ed):
        '''test eos table in the form of interpolating from data stored
        in the 2d image buffer image2d_t'''
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        compile_options = []
        eos_table = self.create_table(ctx, compile_options)

        CL_SOURCE = '''//CL//
        __kernel void interpolate(
            read_only image2d_t src,
            __global float4 * result,
            const float test_ed)
        {
            float ed_per_row = EOS_ED_STEP*EOS_NUM_OF_COLS;
            int row = test_ed/ed_per_row;
            int col = (test_ed - EOS_ED_START - row*ed_per_row)
                       /EOS_ED_STEP;

            const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE
                  | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

            if ( get_global_id(0) == 0 ) {
                float4 result4;
                result4 = read_imagef(src, sampler, (int2)(col, row));
                *result = result4;
            }
        }
        '''

        prg = cl.Program(ctx, CL_SOURCE).build(self.compile_options)

        mf = cl.mem_flags

        h_result = np.empty(4, dtype=self.cfg.real)
        result = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=h_result)
        prg.interpolate(queue, (1,), None, eos_table, result, np.float32(test_ed))
        cl.enqueue_copy(queue, h_result, result)

        print('result at (', test_ed, ')=', h_result[0])
     

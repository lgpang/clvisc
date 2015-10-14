#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import pyopencl as cl
import numpy as np

class Eos(object):
    '''create eos table for hydrodynamic simulation;
    the (ed, pr, T, s) is stored in image2d buffer
    for fast linear interpolation'''
    def __init__(self, cfg, ctx, queue, compile_options):
        # information of the eos table
        ed_start, ed_step, num_of_ed = None, None, None
        if cfg.IEOS == 2:
            import wb
            self.ed = wb.ed
            self.pr = wb.pr
            self.T = wb.T
            ed_start = wb.ed_start
            ed_step = wb.ed_step
            num_of_ed = wb.num_ed
        elif cfg.IEOS == 3:
            import glueball
            self.ed = glueball.ed
            self.pr = glueball.pr
            self.T = glueball.T
            ed_start = glueball.ed_start
            ed_step = glueball.ed_step
            num_of_ed = glueball.num_ed

        self.s = (self.ed + self.pr)/self.T
        self.ctx = ctx
        self.queue = queue

        compile_options.append('-D EOS_ED_START={value}'.format(
                                                 value=ed_start))
        compile_options.append('-D EOS_ED_STEP={value}'.format(
                                                 value=ed_step))
        compile_options.append('-D EOS_NUM_ED={value}'.format(
                                                 value=num_of_ed))

        self.compile_options = compile_options

        
    def create_image2d(self, num_of_rows=200, num_of_cols=1000):
        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
        src = np.array(zip(self.ed, self.pr, self.T, self.s),
                 dtype=np.float32).reshape(num_of_rows, num_of_cols, 4)

        eos_table = cl.image_from_array(self.ctx, src, 4)
        return eos_table, num_of_rows, num_of_cols

    def test_eos(self, cfg, test_ed):
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

        eos_table, nrows, ncols = self.create_image2d()
        self.compile_options.append('-D EOS_NUM_OF_ROWS=%s'%nrows)
        self.compile_options.append('-D EOS_NUM_OF_COLS=%s'%ncols)
        prg = cl.Program(self.ctx, CL_SOURCE).build(self.compile_options)
        print self.compile_options

        mf = cl.mem_flags

        h_result = np.empty(1, dtype=cfg.real4)
        result = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=h_result)
        prg.interpolate(self.queue, (1,), None, eos_table, result, np.float32(test_ed))
        cl.enqueue_copy(self.queue, h_result, result)

        print('result at (', test_ed, ')=', h_result)
     

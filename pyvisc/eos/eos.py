#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import numpy as np
from scipy.interpolate import interp1d

class Eos(object):
    '''create eos table for hydrodynamic simulation;
    the (ed, pr, T, s) is stored in image2d buffer
    for fast linear interpolation'''
    def __init__(self, IEOS=0):
        # information of the eos table
        if IEOS == 0:
            self.ideal_gas()
        elif IEOS == 1:
            self.lattice_pce()
        elif IEOS == 2:
            self.lattice_ce()
        elif IEOS == 3:
            self.pure_su3()

    def ideal_gas(self):
        hbarc = 0.1973269
        dof = 169.0/4.0
        coef = np.pi*np.pi/30.0
        self.f_P = lambda ed: ed/3.0
        self.f_T = lambda ed: hbarc*np.power(1.0/(dof*coef)*ed/hbarc, 0.25) + 1.0E-10
        self.f_S = lambda ed: (ed + self.f_P(ed))/self.f_T(ed)
        self.f_ed = lambda T: dof*coef*hbarc*(T/hbarc)**4.0

    def eos_func_from_interp1d(self):
        # interpolation functions
        self.f_ed = interp1d(self.T, self.ed)
        self.f_T = interp1d(self.ed, self.T)
        self.f_P = interp1d(self.ed, self.pr)
        self.f_S = interp1d(self.ed, self.s)

    def lattice_pce(self):
        import os
        cwd, cwf = os.path.split(__file__)
        pce = np.loadtxt(os.path.join(cwd, 'eos_table/PCE_PST.dat'))
        self.ed = np.insert(0.5*(pce[1:, 0] + pce[:-1, 0]), 0, 0.0)
        self.pr = np.insert(0.5*(pce[1:, 1] + pce[:-1, 1]), 0, 0.0)
        self.s = np.insert(0.5*(pce[1:, 2] + pce[:-1, 2]), 0, 0.0)
        self.T = np.insert(0.5*(pce[1:, 3] + pce[:-1, 3]), 0, 0.0)
        self.ed_start = 0.0
        self.ed_step = 0.002
        self.num_of_ed = 155500
        self.eos_func_from_interp1d()

    def lattice_ce(self):
        '''lattice qcd EOS from wuppertal budapest group
        2014 with chemical equilibrium EOS'''
        import wb
        self.ed = wb.ed
        self.pr = wb.pr
        self.T = wb.T
        T = np.copy(self.T)
        T[T < 1.0E-15] = 1.0E-15
        self.s = (self.ed + self.pr)/T
        self.ed_start = wb.ed_start
        self.ed_step = wb.ed_step
        self.num_of_ed = wb.num_ed
        self.eos_func_from_interp1d()

    def pure_su3(self):
        '''pure su3 gauge EOS'''
        import glueball
        self.ed = glueball.ed
        self.pr = glueball.pr
        self.T = glueball.T
        T = np.copy(self.T)
        T[T < 1.0E-15] = 1.0E-15
        self.s = (self.ed + self.pr)/T
        self.ed_start = glueball.ed_start
        self.ed_step = glueball.ed_step
        self.num_of_ed = glueball.num_ed
        self.eos_func_from_interp1d()

    def create_table(self, ctx, compile_options, nrow=200, ncol=1000):
        '''store the eos (ed, pr, T, s) in image2d_t table for fast
        linear interpolation,
        add some information to compile_options for EOS table'''
        import pyopencl as cl
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
        import pyopencl as cl
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

        h_result = np.empty(4, dtype=np.float32)
        result = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=h_result)
        prg.interpolate(queue, (1,), None, eos_table, result, np.float32(test_ed))
        cl.enqueue_copy(queue, h_result, result)

        print('result at (', test_ed, ')=', h_result[0])
     

if __name__ == '__main__':
    eos = Eos(0)
    print eos.f_ed(0.63)

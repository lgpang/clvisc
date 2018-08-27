#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import numpy as np
from scipy.interpolate import interp1d

# the InterpolatedUnivariateSpline works for both interpolation and extrapolation
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit

def minmod(a, b):
    if a>0 and b>0:
        return min(a, b)
    elif a<0 and b<0:
        return max(a, b)
    else:
        return 0

class Eos(object):
    '''create eos table for hydrodynamic simulation;
    the (ed, pr, T, s) is stored in image2d buffer
    for fast linear interpolation'''
    def __init__(self, eos_type='ideal_gas'):
        if eos_type == 'ideal_gas':
            self.ideal_gas()
        elif eos_type == 'lattice_pce165':
            self.lattice_pce165()
        elif eos_type == 'lattice_pce150':
            self.lattice_pce150()
        elif eos_type == 'lattice_wb':
            self.lattice_ce()
        elif eos_type == 'pure_gauge':
            self.pure_su3()
        elif eos_type == 'first_order':
            self.eosq()



    def ideal_gas(self):
        '''ideal gas eos, P=ed/3 '''
        hbarc = 0.1973269
        dof = 169.0/4.0
        coef = np.pi*np.pi/30.0
        self.f_P = lambda ed: np.array(ed)/3.0
        self.f_T = lambda ed: hbarc*(1.0/(dof*coef)*np.array(ed)/hbarc)**0.25 + 1.0E-10
        self.f_S = lambda ed: (np.array(ed) + self.f_P(ed))/self.f_T(ed)
        self.f_ed = lambda T: dof*coef*hbarc*(np.array(T)/hbarc)**4.0
        self.f_cs2 = lambda ed: 1.0/3.0 * np.ones_like(ed)
        self.ed = np.linspace(0, 1999.99, 200000)
        self.pr = self.f_P(self.ed)
        self.T = self.f_T(self.ed)
        self.s = self.f_S(self.ed)
        self.cs2 = self.f_cs2(self.ed)
        self.ed_start = 0.0
        self.ed_step = 0.01
        self.num_of_ed = 200000

    def eosq(self):
        import eosq
        self.ed = eosq.ed
        self.pr = eosq.pr
        self.T = eosq.T
        self.s = eosq.s
        self.ed_start = eosq.ed_start
        self.ed_step = eosq.ed_step
        self.num_of_ed = eosq.num_ed
        #get cs2 using dp/de
        self.eos_func_from_interp1d()
        self.f_P = eosq.f_P
        self.f_T = eosq.f_T
        self.f_S = eosq.f_S
        self.f_ed = eosq.f_ed


    def eos_func_from_interp1d(self, order=1):
        # construct interpolation functions
        self.f_ed = InterpolatedUnivariateSpline(self.T, self.ed, k=order, ext=0)
        self.f_T = InterpolatedUnivariateSpline(self.ed, self.T, k=order, ext=0)
        self.f_P = InterpolatedUnivariateSpline(self.ed, self.pr, k=order, ext=0)
        self.f_S = InterpolatedUnivariateSpline(self.ed, self.s, k=order, ext=0)
        # calc the speed of sound square
        self.cs2 = np.gradient(self.pr, self.ed_step)
        # remove high gradient in dp/de function
        for i, cs2_ in enumerate(self.cs2):
            if abs(cs2_) > 0.34:
                a = self.cs2[i-1]
                b = cs2_
                c = self.cs2[i+1]
                self.cs2[i] = minmod(a, minmod(b, c))
        mask = self.ed >= 30.
        ed_mask = self.ed[mask]
        cs2_mask = self.cs2[mask]
        def exp_func(x, a, b, c):
            '''fit cs2 at ed >30 with a smooth curve'''
            return a / (np.exp(b/x) + c)
        popt, pcov = curve_fit(exp_func, ed_mask, cs2_mask)
        self.cs2[mask] = exp_func(ed_mask, *popt)

    def lattice_pce165(self):
        import os
        cwd, cwf = os.path.split(__file__)
        pce = np.loadtxt(os.path.join(cwd, 'eos_table/s95p-PCE165-v0/EOS_PST.dat'))
        self.ed = np.insert(0.5*(pce[1:, 0] + pce[:-1, 0]), 0, 0.0)
        self.pr = np.insert(0.5*(pce[1:, 1] + pce[:-1, 1]), 0, 0.0)
        self.s = np.insert(0.5*(pce[1:, 2] + pce[:-1, 2]), 0, 0.0)
        self.T = np.insert(0.5*(pce[1:, 3] + pce[:-1, 3]), 0, 0.0)
        self.ed_start = 0.0
        self.ed_step = 0.002
        self.num_of_ed = 155500
        self.eos_func_from_interp1d()


    def lattice_pce150(self):
        import os
        cwd, cwf = os.path.split(__file__)
        pce = np.loadtxt(os.path.join(cwd, 'eos_table/s95p-PCE-v1/EOS_PST.dat'))
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
        from . import wb
        self.ed = wb.ed
        self.pr = wb.pr
        self.T = wb.T
        self.s = (self.ed + self.pr)/(self.T + 1.0E-10)
        self.ed_start = wb.ed_start
        self.ed_step = wb.ed_step
        self.num_of_ed = wb.num_ed
        self.eos_func_from_interp1d()

    def lattice_ce_mod(self):
        '''lattice qcd EOS from wuppertal budapest group
        2014 with chemical equilibrium EOS
        use T=np.linspace(0.03, 1.13, 1999) to create the table,
        notice that ed_step is not constant'''
        import wb_mod as wb
        self.ed = wb.ed
        self.pr = wb.pr
        self.T = wb.T
        self.s = (self.ed + self.pr)/(self.T + 1.0E-10)
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
        self.s = (self.ed + self.pr)/(self.T + 1.0E-10)
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
        src = np.array(list(zip(self.cs2, self.pr, self.T, self.s)),
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

        prg = cl.Program(ctx, CL_SOURCE).build(' '.join(self.compile_options))

        mf = cl.mem_flags

        h_result = np.empty(4, dtype=np.float32)
        result = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=h_result)
        prg.interpolate(queue, (1,), None, eos_table, result, np.float32(test_ed))
        cl.enqueue_copy(queue, h_result, result)

        print('result at (', test_ed, ')=', h_result[0])
     

   
if __name__ == '__main__':
    def test_plot_cs2():
        eos = Eos('lattice_pce150')
        #print eos.f_ed(0.63)
        #print eos.f_ed(0.137)
        import matplotlib.pyplot as plt
        #print(eos.f_ed(0.137))
        plt.plot(eos.ed, np.gradient(eos.pr, eos.ed_step))
        plt.plot(eos.ed, eos.cs2, 'r-')
        plt.show()
 
    test_plot_cs2()

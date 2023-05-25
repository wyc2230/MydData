
import sys
sys.path.append("D:\\Program Files\\Lumerical\\v202\\api\\python") # 添加至Win默认路径
import lumapi
from collections import OrderedDict # OrderedDict包只适用于Python 2，可以安装适用于Python 3的Collections包
import numpy as np
import copy
import scipy as sp
import matplotlib.pyplot as plt
from loguru import logger
import os
import time

'----------------------------------------------------------------------------------------------------------------------'
from MyParams import my_str as para_str
from MyParams import my_frame as para_frame
from MyParams import my_element as para_element
from MyParams import my_source as para_source
from MyParams import my_sweep as para_sweep
from MyParams import my_metalens as para_let

from MyParams import my_fig1 as para_fig1

'======================================================================================================================'
class QYFDTD(object):
    '''类属性, 类属性是每个实例的共有属性，在整个class有效，相当于全局变量；'''
    '物理参数'
    c = para_source['c'];    # 光速

    '''======================================== 参数设置 ============================================================'''
    def __init__(self, path_file,model_name,lambda_start,lambda_stop,freq_num
                 ,element_rotation_z,element_size_1,element_size_2,element_size_3,element_size_4):
        '''实例属性,实例属性用于区分不同的实例，相当于局部变量;
        可以通过外部输入，方便设置'''

        '''路径定义'''
        self.path_file = path_file;
        self.path_file_scripts = path_file + 'scripts\\';
        self.path_file_temp = path_file + 'temp\\';
        self.path_file_results = path_file + 'results\\';
        this_path_name = 'Elment\\'
        self.path_file_results_models = path_file + 'results\\' + this_path_name + 'models\\';
        self.path_file_results_figs = path_file + 'results\\' + this_path_name + 'figs\\';
        self.path_file_results_data = path_file + 'results\\' + this_path_name + 'data\\';
        '创建目录--如果已有则不重新创建'
        os.makedirs(self.path_file_scripts, mode=0o777, exist_ok=True)
        os.makedirs(self.path_file_results, mode=0o777, exist_ok=True)
        os.makedirs(self.path_file_temp, mode=0o777, exist_ok=True)
        os.makedirs(self.path_file_results_models, mode=0o777, exist_ok=True)
        os.makedirs(self.path_file_results_figs, mode=0o777, exist_ok=True)
        os.makedirs(self.path_file_results_data, mode=0o777, exist_ok=True)

        '''仿真模型的文件名'''
        self.model_name = model_name;  # 这个程序所保存的文件名

        '''光源的参数 '''
        self.lambda_start = lambda_start;  # 入射起始波长
        self.lambda_stop = lambda_stop;    # 入射起始波长
        self.freq_num = freq_num;          # 监视器取的频率数目，也可以实现参数扫描，但是这种方法小数点后面不准
        self.propagation_axis = para_source['propagation_axis'];            # 光源的传播方向平行于哪个轴
        self.propagation_direction = para_source['propagation_direction'];  #  1：Forward；-1：Backward; 用数字便于计算
        self.source_shape = para_source['source_shape'];  # 1：Guassian；2：plane wave；3：cauchy-Lorentizan
        self.polarization_angle_x = para_source['polarization_angle_x']; # 光源的偏振角度 Polarization angle of the source
        self.polarization_angle_y = para_source['polarization_angle_y']; # 光源的偏振角度 Polarization angle of the source
        self.plane_wave_type = para_source['plane_wave_type'];    # 1 for Bloch periodic, 2 for BFAST
        self.angle_theta = para_source['angle_theta']; # 光源的倾斜角度，
        self.angle_phi = para_source['angle_phi']; # 光源的倾斜角度，传播方向 绕着 入射平面法线 右转的夹角
        '''这里定义LCP为y方向的相位落后x方向的相位（phi_y-phi_x=-90°）'''
        self.source_polarization_axis_x = para_source['source_polarization_axis_x'];  # 光源偏振方向与坐标轴的夹角，假设0°为水平偏振，则-90°为逆时针选择90°
        self.source_polarization_axis_y = para_source['source_polarization_axis_y'];  # 光源偏振方向与坐标轴的夹角，假设0°为水平偏振，则-90°为逆时针选择90°

        '''超表面结构的尺寸 '''
        self.element_rotation_z = element_rotation_z;   # 待扫描的参数
        self.element_size_H = element_size_1;           # 待扫描的参数
        self.element_size_L = element_size_2;           # 待扫描的参数
        self.element_size_W = element_size_3;           # 待扫描的参数
        self.element_size_r1 = element_size_2;          # 待扫描的参数
        self.element_size_r2 = element_size_3;          # 待扫描的参数
        self.element_arc_theta = element_size_4;        # 待扫描的参数
        self.film_d = para_frame['film_d'];             # 薄膜的厚度=超表面单元的高度
        self.element_bottom = para_frame['element_bottom']; # 超表面element和substrate的界面 z=0
        self.sub_d = para_frame['sub_d'];               # 基底的厚度
        self.sub_z_min = para_frame['sub_z_min'];       # 基底的厚度
        self.z_interface = para_frame['z_interface'];   # 超表面element和substrate的界面 z=0
        self.z_fdtd_min = para_frame['z_fdtd_min'];     # 超表面element和substrate的界面 z=0
        '''超表面单元和FDTD的Unit '''
        self.Ux = para_frame['Ux'];     # x方向的长度：单元边长 特征尺寸 衬底尺寸
        self.Uy = para_frame['Uy'];     # x方向的长度：单元边长 特征尺寸 衬底尺寸

        '''monitor的位置 '''
        self.plane_offset = para_frame['plane_offset'];
        # self.plane_offset = self.plane_offset;
        self.source_z = self.element_size_H + 3*self.plane_offset;          # 用于调试 S参数
        # self.focal_z_center = 0.5*(self.z_interface + self.z_fdtd_min);    # 用于观察焦点
        # self.focal_z_span = abs(self.z_interface - self.z_fdtd_min);       # 用于观察焦点

        'z_R point = index = element_H + 2*plane_offset; plane = point - 2*source_offset'
        self.monitor_plane_R_z = self.source_z - self.plane_offset;  # 距离模型的顶端（element的上表面）1倍间隔
        self.monitor_index_R_z = self.source_z + self.plane_offset;  # 距离模型的顶端（element的上表面）3倍间隔
        self.monitor_point_R_z = self.source_z + self.plane_offset;  # 距离模型的顶端（element的上表面）3倍间隔

        'z_T plane = index =point'
        self.monitor_plane_T_z = self.sub_z_min - 1 * self.plane_offset;  # 距离模型的底端（substrate的下表面）1倍间隔
        self.monitor_index_T_z = self.sub_z_min - 1 * self.plane_offset;  # 距离模型的底端（substrate的下表面）1倍间隔
        self.monitor_point_T_z = self.sub_z_min - 1 * self.plane_offset;  # 距离模型的底端（substrate的下表面）1倍间隔

        'z_in plane = index =point， 相对于样品一个 plane_offset， 理论上和反射plane同位置'
        self.monitor_plane_in_z = self.element_size_H + self.plane_offset;  # 距离模型的顶端（element的上表面）100nm
        self.monitor_index_in_z = self.element_size_H + self.plane_offset;  # 距离模型的顶端（element的上表面）100nm
        self.monitor_point_in_z = self.element_size_H + self.plane_offset;  # point = plane

        'FDTD的区域'
        '''FDTD仿真区域的z_max=monitor_point_R_z，反射 点 监视器）
           FDTD仿真区域的z_min=monitor_plane_T_z，透射 监视器）'''
        # self.FDTD_offset = 0.5 * (self.monitor_point_R_z + self.monitor_plane_T_z); # 注意正负号
        # self.FDTD_len = abs(self.monitor_point_R_z - self.monitor_plane_T_z);       # monitor两端的距离
        # self.FDTD_len_span = self.FDTD_len + 2 * self.plane_offset ;             # 略大于 FDTD两端的距离
        # self.FDTD_len_center = self.z_interface + self.FDTD_offset;

        self.z_fdtd_max = self.monitor_point_R_z;
        self.z_fdtd_min = self.z_fdtd_min;
        self.z_meta_max = self.element_size_H;
        self.z_meta_min = self.sub_z_min;

        self.mesh_fdtd_dx = para_frame['mesh_fdtd_dx'];               # 自定义mesh的精度
        self.mesh_fdtd_dy = para_frame['mesh_fdtd_dy'];               # 自定义mesh的精度
        self.mesh_meta_dx = para_frame['mesh_meta_dx'];               # 自定义mesh的精度
        self.mesh_meta_dy = para_frame['mesh_meta_dy'];               # 自定义mesh的精度
        self.mesh_accuracy = para_frame['mesh_accuracy'];   # FDTD 仿真的网格精度，值越小，网格越粗，速度快，易调试


    '''========================================= 建立模型计算数据 ====================================================='''
    def build_structure(self):
        self.t1_build_structure = time.time()
        with lumapi.FDTD(hide = True) as fdtd:
           '''---------------------------- model>>Structures ------------------------------------------------'''
           '''设置 衬底 substrate'''
           props_Substrate = OrderedDict([("name", "Substrate")
                                             ,("x", 0.0e-9)
                                             ,("x span", self.Ux)
                                             ,("y", 0.0e-9)
                                             ,("y span", self.Uy)
                                             ,("z max", 0.0e-9)
                                             ,("z min", self.sub_z_min)
                                             ,("material", para_frame['material_sub'])
                                          ])

           '''设置 微纳单元 Element'''
           props_Element_rect1 = OrderedDict([("name", "Element_rect01")
                                             , ("x", 0.0e-9)
                                             , ("x span", self.element_size_W)
                                             , ("y", 0.0e-9)
                                             , ("y span", self.element_size_L)
                                             , ("z max", self.element_size_H)
                                             , ("z min", self.element_bottom)
                                             , ("material", para_frame['material_element'])
                                              ]) # 超表面单元
           props_Element_rect2a = OrderedDict([("name", "Element_rect01")
                                             ,("x", 0.0e-9)
                                             ,("x span", self.element_size_W)
                                             ,("y", 0.0e-9)
                                             ,("y span", self.element_size_L)
                                             ,("z max", self.element_size_H)
                                             ,("z min", self.element_bottom)
                                             ,("material", para_frame['material_etch'])
                                          ]) # etch 超表面单元
           props_Element_rect2b = OrderedDict([("name", "Element_rect01")
                                                  , ("x", 0.0e-9)
                                                  , ("x span", self.Ux)
                                                  , ("y", 0.0e-9)
                                                  , ("y span", self.Uy)
                                                  , ("z max", self.element_size_H)
                                                  , ("z min", self.element_bottom)
                                                  , ("material", para_frame['material_element'])
                                               ]) # 矩形薄膜
           props_Element_ring01 = OrderedDict([
                                       ("name", "Element_ring01")
                                       , ("x", 0.0e-9)
                                       , ("y", 0.0e-9)
                                       , ("z max", self.element_size_H)
                                       , ("z min", self.element_bottom)
                                       , ("outer radius", self.element_size_r1)
                                       , ("inner radius", self.element_size_r2)
                                       , ("theta start", 0)
                                       , ("theta stop", self.element_arc_theta)
                                       , ("material", para_frame['material_element'])
                                         ]) # 完整圆环
           props_Element_ring02 = OrderedDict([
                                       ("name", "Element_ring02")
                                       , ("x", 0.0e-9)
                                       , ("y", 0.0e-9)
                                       , ("z max", self.element_size_H)
                                       , ("z min", self.element_bottom)
                                       , ("outer radius", self.element_size_r1)
                                       , ("inner radius", self.element_size_r2)
                                       , ("theta start", 0)
                                       , ("theta stop", self.element_arc_theta)
                                       , ("material", para_frame['material_element'])
                                         ]) # 缺口

           '''---------------------------- model>>Simulation ------------------------------------------------'''
           '''设置 仿真区域 '''
           props_fdtd = OrderedDict([
               ("dimension", 2),  #
               ("x", 0.0e-9),
               ("x span", self.Ux),
               ("y", 0.0e-9),
               ("y span", self.Uy),
               ("z max", self.z_fdtd_max),
               ("z min", self.z_fdtd_min),
               ("mesh accuracy", self.mesh_accuracy),
               ("x min bc", 3),  #边界条件 3：Periodic
               ("y min bc", 3),  #
               ("z min bc", 1),  #边界条件 1：PML
               ("z max bc", 1),  #
           ])

           '''设置 光源'''
           props_Source_x = OrderedDict([
               ("name", "Source_x"),
               ("injection axis", self.propagation_axis),  #
               ("direction", 'Backward'),  #
               ("source shape", self.source_shape),
               ("phase", self.source_polarization_axis_x),
               ("polarization angle", self.polarization_angle_x),
               ("plane wave type", self.plane_wave_type), # 1 for Bloch periodic, 2 for BFAST
               # ("angle theta", self.angle_theta), # periodic 没有角度选项
               # ("angle phi", self.angle_phi),     # periodic 没有角度选项
               ("x", 0.0e-9),
               ("x span", self.Ux),
               ("y", 0.0e-9),
               ("y span", self.Uy),
               ("z", self.source_z),
               ("wavelength start", self.lambda_start),
               ("wavelength stop", self.lambda_stop),
           ])
           props_Source_y = OrderedDict([
               ("name", "Source_y"),
               ("injection axis", self.propagation_axis),  #
               ("direction", 'Backward'),  #
               ("source shape", self.source_shape),
               ("phase", self.source_polarization_axis_y),
               ("polarization angle", self.polarization_angle_y),
               # ("source type", self.source_type), # 1 for Bloch periodic, 2 for BFAST（plane wave type）
               # ("angle theta", self.angle_theta), # periodic 没有角度选项
               # ("angle phi", self.angle_phi),     # periodic 没有角度选项
               ("x", 0.0e-9),
               ("x span", self.Ux),
               ("y", 0.0e-9),
               ("y span", self.Uy),
               ("z", self.source_z),
               ("wavelength start", self.lambda_start),
               ("wavelength stop", self.lambda_stop),
           ])

           ''' 设置 网格 mesh - 仿真计算的精度'''
           props_mesh_fdtd = OrderedDict([
               ("name", "mesh_fdtd"),
               ("dx", self.mesh_fdtd_dx),  #
               ("dy", self.mesh_fdtd_dx),  #
               ("x", 0.0e-9),
               ("y", 0.0e-9),
               ("x span", self.Ux),
               ("y span", self.Uy),
               ("z max", self.z_fdtd_max),
               ("z min", self.z_fdtd_min),
                ])
           props_mesh_meta = OrderedDict([
               ("name", "mesh_meta"),
               ("dx", self.mesh_meta_dx),  #
               ("dy", self.mesh_meta_dy),  #
               ("x", 0.0e-9),
               ("y", 0.0e-9),
               ("x span", self.Ux),
               ("y span", self.Uy),
               ("z max", self.z_meta_max),
               ("z min", self.z_meta_min),
                ])

           '''---------------------------- model>>monitors ------------------------------------------------'''
           '''设置 监视器 -- 面监视器 --------'''
           props_T_plane = OrderedDict([
               ("name", "T_plane"),
               ("override global monitor settings", 1), # ture=1=uniform
               ("monitor type", '2D z-normal'),  # 7:2D z-normal
               ("frequency points", self.freq_num),
               ("x", 0.0e-9),
               ("x span", self.Ux),
               ("y", 0.0e-9),
               ("y span", self.Uy),
               ("z", self.monitor_plane_T_z),
                ])
           props_R_plane = OrderedDict([
               ("name", "R_plane"),
               ("override global monitor settings", 1),  # ture=1=uniform
               ("monitor type", '2D z-normal'),  # 7:2D z-normal
               ("frequency points", self.freq_num),
               ("x", 0.0e-9),
               ("x span", self.Ux),
               ("y", 0.0e-9),
               ("y span", self.Uy),
               ("z", self.monitor_plane_R_z),
           ])
           props_in_plane = OrderedDict([
               ("name", "in_plane"),
               ("override global monitor settings", 1),  # ture=1=uniform
               ("monitor type", '2D z-normal'),  # 7:2D z-normal
               ("frequency points", self.freq_num),
               ("x", 0.0e-9),
               ("x span", self.Ux),
               ("y", 0.0e-9),
               ("y span", self.Uy),
               ("z", self.monitor_plane_in_z),
           ])

           '''设置 监视器 -- 点监视器 -----------'''
           props_T_point = OrderedDict([
               ("name", "T_point"),
               ("monitor type", 'Point'),  # 1: Point
               ("override global monitor settings", 1), # ture=1=uniform
               ("frequency points", self.freq_num),
               ("x", 0.),
               ("y", 0.),
               ("z", self.monitor_point_T_z),
           ])
           props_R_point = OrderedDict([
               ("name", "R_point"),
               ("monitor type", 'Point'),  # 1: Point
               ("override global monitor settings", 1), # ture=1=uniform
               ("frequency points", self.freq_num),
               ("x", 0.),
               ("y", 0.),
               ("z", self.monitor_point_R_z),
           ])
           props_in_point = OrderedDict([
               ("name", "in_point"),
               ("monitor type", 'Point'),  # 1: Point
               ("override global monitor settings", 1), # ture=1=uniform
               ("frequency points", self.freq_num),
               ("x", 0.),
               ("y", 0.),
               ("z", self.monitor_point_in_z),
           ])

           '''设置 S参数index监视器 ------------'''
           props_T_index = OrderedDict([
               ("name", "T_index"),
               ("monitor type", '3D'),  #
               ("spatial interpolation", 'nearest mesh cell'),  #
               ("x", 0.),
               ("y", 0.),
               ("z", self.monitor_index_T_z),
               ("x span", 0),
               ("y span", 0),
               ("z span", 0),
           ])
           props_R_index = OrderedDict([
               ("name", "R_index"),
               ("monitor type", '3D'),  #
               ("spatial interpolation", 'nearest mesh cell'),  #
               ("x", 0.),
               ("y", 0.),
               ("z", self.monitor_index_R_z),
               ("x span", 0),
               ("y span", 0),
               ("z span", 0),
           ])
           props_in_index = OrderedDict([
               ("name", "in_index"),
               ("monitor type", '3D'),  #
               ("spatial interpolation", 'nearest mesh cell'),  #
               ("x", 0.),
               ("y", 0.),
               ("z", self.monitor_index_in_z),
               ("x span", 0),
               ("y span", 0),
               ("z span", 0),
           ])

           props_T_xz_plane = OrderedDict([
               ("name", "T_xz_plane"),
               ("override global monitor settings", 1), # ture=1=uniform
               ("monitor type", '2D Y-normal'),  # 7:2D z-normal
               ("frequency points", self.freq_num),
               ("x", 0.0e-9),
               ("x span", self.Ux),
               ("y", 0.0e-9),
               ("z max", self.z_interface),
               ("z min", self.z_fdtd_min),
                ])

           props_T_yz_plane = OrderedDict([
               ("name", "T_yz_plane"),
               ("override global monitor settings", 1),  # ture=1=uniform
               ("monitor type", '2D X-normal'),  # 7:2D z-normal
               ("frequency points", self.freq_num),
               ("x", 0.0e-9),
               ("y", 0.0e-9),
               ("y span", self.Uy),
               ("z max", self.monitor_point_R_z),
               ("z min", self.z_fdtd_min),
                ])

           '''================================== script =========================================================='''
           '添加 材料'
           fdtd.importmaterialdb(self.path_file_scripts+"GaN - PRB.mdf");

           '添加 基底'
           fdtd.addrect(properties=props_Substrate)

           '添加 超表面结构 带有缺口的圆环'
           # fdtd.addring(properties=props_Element_ring01)
           # fdtd.addring(properties=props_Element_ring02)
           # '''旋转 超表面结构'''
           # fdtd.setnamed('Element_ring02','first axis', 'z')
           # fdtd.setnamed('Element_ring02','rotation 1', self.element_rotation_z);  # [degrees]
           '添加 超表面结构 rect'
           fdtd.addrect(properties=props_Element_rect1)
           '''旋转 超表面结构'''
           fdtd.setnamed('Element_rect01','first axis', 'z')
           fdtd.setnamed('Element_rect01','rotation 1', self.element_rotation_z);  # [degrees]

           '添加 光源 x 偏振'
           fdtd.addplane(properties=props_Source_x)
           '添加 光源 y 偏振'
           fdtd.addplane(properties=props_Source_y)

           '添加 fdtd仿真区域 以及 自定义的网格'
           fdtd.addfdtd(properties=props_fdtd)
           fdtd.addmesh(properties=props_mesh_fdtd)
           fdtd.addmesh(properties=props_mesh_meta)

           '添加 面监视器'
           fdtd.addpower(properties=props_T_plane)
           fdtd.addpower(properties=props_R_plane)
           fdtd.addpower(properties=props_in_plane)
           fdtd.addpower(properties=props_T_xz_plane)
           fdtd.addpower(properties=props_T_yz_plane)
           '添加 点监视器'
           fdtd.addpower(properties=props_T_point)
           fdtd.addpower(properties=props_R_point)
           fdtd.addpower(properties=props_in_point)
           '添加 index监视器'
           fdtd.addindex(properties=props_T_index)
           fdtd.addindex(properties=props_R_index)
           fdtd.addindex(properties=props_in_index)

           '''保存模型'''
           self.str_filename = self.path_file_results_models + self.model_name + ".fsp"
           fdtd.save(self.str_filename);


    '''========================================= 建立模型计算数据 ====================================================='''
    def run_model(self):
        self.t1_run_model = time.time()
        with lumapi.FDTD(filename=self.str_filename,hide = True) as fdtd:
           '''开始仿真'''
           fdtd.run()

           '''读取监视器中的数据-------------------------------------------------'''
           '''用getdata( )函数可以获取监视器的原始数据，注意与getresult( )区分'''
           m_T_plane = "T_plane";
           self.f = np.array(fdtd.getdata(m_T_plane, "f"));
           self.x = fdtd.getdata(m_T_plane, "x");
           self.y = fdtd.getdata(m_T_plane, "y");
           self.Ex = np.squeeze(fdtd.getdata(m_T_plane, "Ex"));
           self.Ey = np.squeeze(fdtd.getdata(m_T_plane, "Ey"));
           self.Hx = np.squeeze(fdtd.getdata(m_T_plane, "Hx"));
           self.Hy = np.squeeze(fdtd.getdata(m_T_plane, "Hy"));
           self.x_lambda = self.c / self.f * 1e9;  # [nm]

           m_T_point = "T_point";
           self.f_point = fdtd.getdata(m_T_point, "f");
           self.x_point = fdtd.getdata(m_T_point, "x");
           self.y_point = fdtd.getdata(m_T_point, "y");
           self.Ex_point = np.squeeze(fdtd.getdata(m_T_point, "Ex"));
           self.Ey_point = np.squeeze(fdtd.getdata(m_T_point, "Ey"));
           self.Hx_point = np.squeeze(fdtd.getdata(m_T_point, "Hx"));
           self.Hy_point = np.squeeze(fdtd.getdata(m_T_point, "Hy"));

           self.Ex_point = np.array([self.Ex_point]);
           self.Ey_point = np.array([self.Ey_point]);
           self.Hx_point = np.array([self.Hx_point]);
           self.Hy_point = np.array([self.Hy_point]);

           m_T_index = "T_index";
           self.f_index = fdtd.getdata(m_T_index, "f");
           self.x_index = fdtd.getdata(m_T_index, "x");
           self.y_index = fdtd.getdata(m_T_index, "y");
           self.index_x = np.squeeze(fdtd.getdata(m_T_index, "index_x"));
           self.index_y = np.squeeze(fdtd.getdata(m_T_index, "index_y"));
           self.index_z = np.squeeze(fdtd.getdata(m_T_index, "index_z"));
           self.t2_base_structure = time.time()


           m_in_plane = "in_plane";
           self.x_in = fdtd.getdata(m_T_plane, "x");
           self.y_in = fdtd.getdata(m_T_plane, "y");
           self.Ex_in = np.squeeze(fdtd.getdata(m_T_plane, "Ex"));
           self.Ey_in = np.squeeze(fdtd.getdata(m_T_plane, "Ey"));
           self.Hx_in = np.squeeze(fdtd.getdata(m_T_plane, "Hx"));
           self.Hy_in = np.squeeze(fdtd.getdata(m_T_plane, "Hy"));


    '''======================================== 保存数据 ============================================================'''
    def save_fdtd_raw_data(self):
        self.t1_save_fdtd_raw_data = time.time()
        np.save(self.path_file_temp + 'f.npy', self.f)
        np.save(self.path_file_temp + 'x.npy', self.x)
        np.save(self.path_file_temp + 'y.npy', self.y)
        np.save(self.path_file_temp + 'Ex.npy', self.Ex)
        np.save(self.path_file_temp + 'Ey.npy', self.Ey)
        np.save(self.path_file_temp + 'Hx.npy', self.Hx)
        np.save(self.path_file_temp + 'Hy.npy', self.Hy)
        np.save(self.path_file_temp + 'x_lambda.npy', self.x_lambda)

        np.save(self.path_file_temp + 'f_point.npy', self.f_point)
        np.save(self.path_file_temp + 'x_point.npy', self.x_point)
        np.save(self.path_file_temp + 'y_point.npy', self.y_point)
        np.save(self.path_file_temp + 'Ex_point.npy', self.Ex_point)
        np.save(self.path_file_temp + 'Ey_point.npy', self.Ey_point)
        np.save(self.path_file_temp + 'Hx_point.npy', self.Hx_point)
        np.save(self.path_file_temp + 'Hy_point.npy', self.Hy_point)

        np.save(self.path_file_temp + 'f_index.npy', self.f_index)
        np.save(self.path_file_temp + 'x_index.npy', self.x_index)
        np.save(self.path_file_temp + 'y_index.npy', self.y_index)
        np.save(self.path_file_temp + 'index_x.npy', self.index_x)
        np.save(self.path_file_temp + 'index_y.npy', self.index_y)
        np.save(self.path_file_temp + 'index_z.npy', self.index_z)

        np.save(self.path_file_temp + 'x_in.npy', self.x_in)
        np.save(self.path_file_temp + 'y_in.npy', self.y_in)
        np.save(self.path_file_temp + 'Ex_in.npy', self.Ex_in)
        np.save(self.path_file_temp + 'Ey_in.npy', self.Ey_in)
        np.save(self.path_file_temp + 'Hx_in.npy', self.Hx_in)
        np.save(self.path_file_temp + 'Hy_in.npy', self.Hy_in)



    '''======================================== S 参数 ============================================================'''
    def calculate_S_parameters(self):
        self.t1_calculate_S_parameters = time.time()
        '用S参数描述效率'
        with lumapi.FDTD(filename=self.str_filename, hide = True) as fdtd:
            self.source_position = fdtd.getdata("Source_x", "z");
            self.position_offset_forward = fdtd.getdata("T_plane", "z");
            self.position_offset_backward = fdtd.getdata("R_plane", "z");
            self.metamaterial_center = 0.5 * (self.element_size_H + self.sub_z_min); # sub_z<0
            self.metamaterial_span = abs(self.element_size_H - self.sub_z_min);
            'source offset: 光源和反射监视器R之间的距离。'
            '-----------------------------------------------------------------------------------------------------'
            self.show_diagnostics = 0;  # enable for debugging
            self.warnings = 0;  # warning counter
            self.small_number = 5e-7;  # small number for error checking
            self.dim = int(fdtd.getdata("R_plane", "dimension"));  # Dimension of simulation
            self.nf = int(self.f.size);  # Frequency array
            # self.nf = int(self.freq_num);  # Frequency array

            'Refractive indices'
            self.refractive_index_r = fdtd.getdata("R_index", "index_z").real;  # reflection
            self.refractive_index_t = fdtd.getdata("T_index", "index_z").real;  # transmission

            '''TOTAL TRANSMITTED AND REFLECTED POWERS:
            positive number. Since monitor "R" is in front of the source, 
            abs(transmission(R)) + R = 1 => R = 1 - abs(transmission(R)) '''
            self.T = fdtd.transmission("T_plane") * self.propagation_direction;  # transmission
            self.R = (1 - fdtd.transmission("R_plane") * self.propagation_direction);  # reflection

            'Magnitude of wavevecktors'
            self.k1 = 2 * np.pi * self.refractive_index_r * self.f / self.c;
            self.k2 = 2 * np.pi * self.refractive_index_t * self.f / self.c;
            self.k1 = np.squeeze(self.k1)
            self.k2 = np.squeeze(self.k2)
            "Components of the wavevectors along the propagation direction:"
            self.k1_prop = fdtd.matrix(self.nf);
            self.k2_prop = fdtd.matrix(self.nf);
            "S-parameters"
            self.S11_Gn = np.empty(self.nf, dtype=complex);
            self.S21_Gn = np.empty(self.nf, dtype=complex);
            self.S11_pol_Gn = np.empty([self.nf, 2], dtype=complex);
            self.S21_pol_Gn = np.empty([self.nf, 2], dtype=complex);

            self.Tn_forward = fdtd.matrix(self.nf);
            self.Tn_backward = fdtd.matrix(self.nf);

            self.source_warning = "false";
            self.targetn_warning = "false";
            self.input_debug = fdtd.matrix(self.nf);

            '-----------------------------------------------------------------------------------------------------'
            "#loop over frequencies"
            print("有问题待解决: target_grating_order_out = 0  # 指定用于计算S参数的特定输出光栅顺序；那么应该是多少呢？" )
            target_grating_order_out = 0  # 指定用于计算S参数的特定输出光栅顺序
            d_address = -1; # FDTD给出的数组第一个地址是1，python的地址第一个是0，差了个1
            py_1 = 1 + d_address;
            py_2 = 2 + d_address + 0;  #
            print("有问题待解决: py_3 = 3 + d_address + 1;  # 不加1数目少一个" )
            py_3 = 3 + d_address + 1;  # 不加1数目少一个

            for fdtd_i in range(1, self.nf):
                py_i = fdtd_i - 1
                print("\nloop over frequencies，fdtd_i = ", fdtd_i
                      ,"， loop over frequencies，py_i = ", py_i)

                "Transmission in far field (plane monitor T_plane):-----------------------------------------------------------"
                n_array = np.array([fdtd.gratingn("T_plane", fdtd_i)]);  # grating order numbers
                self.G_forward = fdtd.gratingpolar("T_plane", fdtd_i, self.refractive_index_t,
                                                   self.propagation_direction);

                if self.dim == 3:
                    m_array = fdtd.gratingm("T_plane", fdtd_i);  # grating order numbers
                    u1 = fdtd.gratingu1("T_plane", fdtd_i);
                    u2 = fdtd.gratingu2("T_plane", fdtd_i);
                    U1 = fdtd.meshgridx(u1, u2);
                    U2 = fdtd.meshgridy(u1, u2);
                    temp = np.array([[np.sqrt(1 - U1 ** 2 - U2 ** 2)]]);
                    py_targetn_index = int(fdtd.find(n_array, target_grating_order_out) + d_address);  #
                    py_targetm_index = int(fdtd.find(m_array, 0) + d_address);  # only scattering in plane of incidence

                    if n_array[py_targetn_index] == target_grating_order_out:
                        print("True in n_array[targetn_index] == target_grating_order_out: ")
                        self.k2_prop[py_i] = self.k2[py_i] * fdtd.pinch(temp[py_targetn_index, py_targetm_index]);
                        self.G_forward = fdtd.pinch(self.G_forward[py_targetn_index, py_targetm_index, py_1:py_3]);
                    else:
                        print("False in n_array[targetn_index] == target_grating_order_out: ")
                        self.targetn_warning = "true";
                        self.G_forward = fdtd.matrix(3);  # return 0

                else:
                    u1 = fdtd.gratingu1("T_plane", fdtd_i);
                    temp = np.array([np.sqrt(1 - u1 ** 2)]);
                    print("False in self.dim == 3 ： temp =", type(temp), temp)
                    py_targetn_index = int(fdtd.find(n_array, target_grating_order_out) + d_address);  #
                    if n_array[py_targetn_index] == target_grating_order_out:
                        print("True in n_array[targetn_index] == target_grating_order_out: ")
                        print("targetn_index = ", type(py_targetn_index), py_targetn_index)
                        self.k2_prop[py_i] = self.k2[py_i] * fdtd.pinch(temp[py_targetn_index]);
                        self.G_forward = fdtd.pinch(self.G_forward[py_targetn_index, py_1:py_3]);
                    else:
                        print("False in n_array[targetn_index] == target_grating_order_out: ")
                        self.targetn_warning = "true";
                        self.G_forward = fdtd.matrix(3);  # return 0

                "Reflection in far field (monitor R):-------------------------------------------------------------"
                n_array = np.array([fdtd.gratingn("R_plane", fdtd_i)]);  # grating order numbers
                self.G_backward = fdtd.gratingpolar("R_plane", fdtd_i, self.refractive_index_r,
                                                    -1 * self.propagation_direction);
                if self.dim == 3:
                    print("Reflection, True in self.dim == 3: ")
                    m_array = fdtd.gratingm("R_plane", fdtd_i);
                    u1 = fdtd.gratingu1("R_plane", fdtd_i);
                    u2 = fdtd.gratingu2("R_plane", fdtd_i);
                    U1 = fdtd.meshgridx(u1, u2);
                    U2 = fdtd.meshgridy(u1, u2);
                    temp = np.array([[np.sqrt(1 - U1 ** 2 - U2 ** 2)]]);
                    py_targetn_index = int(fdtd.find(n_array, target_grating_order_out) + d_address);  #
                    py_targetm_index = int(fdtd.find(m_array, 0) + d_address);  # only scattering in plane of incidence

                    if n_array[py_targetn_index] == target_grating_order_out:
                        print("Reflection, True in n_array[targetn_index] == target_grating_order_out: ")
                        self.k2_prop[py_i] = self.k2[py_i] * fdtd.pinch(temp[py_targetn_index, py_targetm_index]);
                        print('self.G_backward : ',self.G_backward.shape,self.G_backward)
                        self.G_backward = fdtd.pinch(self.G_backward[py_targetn_index, py_targetm_index, py_1:py_3]);
                        print('self.G_backward : ',self.G_backward.shape,self.G_backward)
                    else:
                        print("Reflection, False in n_array[targetn_index] == target_grating_order_out: ")
                        self.targetn_warning = "true";
                        self.G_backward = fdtd.matrix(3);  # return 0

                else:
                    print("Reflection, False in self.dim == 3: ")
                    u1 = fdtd.gratingu1("R_plane", fdtd_i);
                    temp = np.array([[np.sqrt(1 - u1 ** 2)]]);
                    py_targetn_index = int(fdtd.find(n_array, target_grating_order_out) + d_address);  #
                    if n_array[py_targetn_index] == target_grating_order_out:
                        self.k2_prop[py_i] = self.k2[py_i] * fdtd.pinch(temp[py_targetn_index]);
                        self.G_backward = fdtd.pinch(self.G_backward[py_targetn_index, py_1:py_3]);
                    else:
                        self.targetn_warning = "true";
                        self.G_backward = fdtd.matrix(3);  # return 0

                "Input in far field (monitor R):------------------------------------------------------------------"
                n_array = np.array([fdtd.gratingn("R_plane", fdtd_i)]);  # grating order numbers
                m_array = np.array([fdtd.gratingm("R_plane", fdtd_i)]);
                self.G_input = fdtd.gratingpolar("R_plane", fdtd_i, self.refractive_index_r,
                                                 self.propagation_direction);
                py_targetn_index = int(fdtd.find(n_array, target_grating_order_out) + d_address);  #
                py_targetm_index = int(fdtd.find(m_array, 0) + d_address);  # only scattering in plane of incidence

                if self.dim == 3:
                    self.G_input = fdtd.pinch(self.G_input[py_targetn_index, py_targetm_index, py_1:py_3]);

                else:
                    self.G_input = fdtd.pinch(self.G_input[py_targetn_index, py_1:py_3]);
                '-------------------------------------------------------------------------------------------------------'
                'Fraction of power to desired order (sum absolute value squared of all three field components)----'
                self.Tn_forward[py_i] = sum(abs(self.G_forward) ** 2);
                self.Tn_backward[py_i] = sum(abs(self.G_backward) ** 2);

                'Additional minus sign for p polarization (Uphi) due to our far-field convention:'
                print('有问题待解决: 去过冗余之后，矩阵会改变顺序')
                # self.G_backward[3 + d_address] = -self.G_backward[3 + d_address];
                self.G_backward[3 + d_address,:] = -self.G_backward[3 + d_address,:];

                'Select largest input field component (s or p) for S-parameter extraction'
                self.tempG = abs(self.G_input) ** 2;  # absolute value squared of three field components
                self.norm_input = np.sqrt(sum(self.tempG));  # for normalization (this corrects for any artificial power going to non-zero grating orders)
                self.pol_to_select = int(fdtd.find(self.tempG, max(self.tempG)) + d_address);

                'Warn user if input is not clean s or p polarization'
                self.testGinput = fdtd.pinch(self.G_input[self.pol_to_select])
                self.tempG = copy.deepcopy(self.G_input)
                self.tempG[self.pol_to_select] = 0;
                if max(abs(self.tempG)) > 1e-4 * min(abs(self.testGinput)):
                    self.source_warning = "true";

                '''Save s (Uphi) and p (Utheta) components of transmission and reflection grating projections, 
                note reversal of order to make s first polarization '''
                self.G_forward_pol = self.G_forward[2 + d_address: 3 + d_address, 0];
                self.G_backward_pol = self.G_backward[2 + d_address: 3 + d_address, 0];
                self.G_forward_pol = np.flip(self.G_forward_pol)
                self.G_backward_pol = np.flip(self.G_backward_pol)

                "Save dominant polarization component of all grating projections"
                self.G_input = fdtd.pinch(self.G_input[self.pol_to_select]) / self.norm_input;
                self.G_forward = fdtd.pinch(self.G_forward[self.pol_to_select]);
                self.G_backward = fdtd.pinch(self.G_backward[self.pol_to_select]);

                " Phase corrections associated with dimension"
                self.phase_correction = 1j;
                if self.dim == 2:
                    self.phase_correction = np.exp(1j * np.pi / 4);  # 2D and 3D are different

                '''Roll back phase to monitor plane for transmission and reflection grating projections and back to
                source for input grating projection'''
                self.r_origin_to_input = self.propagation_direction * 1.0 - 0.0;
                self.r_source_to_origin = 0.0 - self.source_position;

                self.phase_source_to_input = self.propagation_direction * self.k1[py_i] * self.r_origin_to_input \
                                             + self.propagation_direction * self.k1_prop[
                                                 py_i] * self.r_source_to_origin;

                self.input = self.G_input * self.phase_correction * np.exp(-1j * self.phase_source_to_input);

                self.r_origin_to_backward = - self.propagation_direction * 1.0 - 0.0;
                self.r_R_to_origin = 0.0 - self.position_offset_backward;

                self.phase_R_to_backward = - self.propagation_direction * self.k1[py_i] * self.r_origin_to_backward \
                                           + (- self.propagation_direction) * self.k1_prop[py_i] * self.r_R_to_origin;

                self.S11_Gn[py_i] = self.G_backward * self.phase_correction \
                                    * np.exp(-1j * self.phase_R_to_backward) / self.input;  # S11 from source to monitor R
                self.S11_pol_Gn[py_i, py_1: py_2] = self.G_backward_pol[py_1: py_2] * self.phase_correction \
                                                    * np.exp(-1j * self.phase_R_to_backward) / self.input;  # S11 from source to monitor R

                self.r_origin_to_forward = self.propagation_direction * 1.0 - 0.0;
                self.r_T_to_origin = 0.0 - self.position_offset_forward;
                self.phase_T_to_forward = self.propagation_direction * self.k2[py_i] * self.r_origin_to_forward \
                                          + self.propagation_direction * self.k2_prop[py_i] * self.r_T_to_origin;

                self.S21_Gn[py_i] = self.G_forward * self.phase_correction \
                                    * np.exp(-1j * self.phase_T_to_forward) \
                                    / self.input;  # S21 from source to monitor T
                self.S21_pol_Gn[py_i, py_1: py_2] = self.G_forward_pol[py_1: py_2] * self.phase_correction \
                                    * np.exp(-1j * self.phase_T_to_forward)\
                                    / self.input;  # S21 from source to monitor T

                'Transmission to desired order'
                self.Tn_forward[py_i] = self.Tn_forward[py_i] * self.T[py_i];
                self.Tn_backward[py_i] = self.Tn_backward[py_i] * self.R[py_i];

                'Normalize S-parameters to source power'
                self.S11_Gn[py_i] = self.S11_Gn[py_i] * np.sqrt(self.R[py_i]);
                self.S21_Gn[py_i] = self.S21_Gn[py_i] * np.sqrt(self.T[py_i]) \
                                    * (np.sqrt(self.refractive_index_r / self.refractive_index_t));

                'correct amplitude if top/bottom substrates are different'
                self.S11_pol_Gn[py_i, py_1: py_2] = self.S11_pol_Gn[py_i, py_1: py_2] * np.sqrt(self.R[py_i]);
                self.S21_pol_Gn[py_i, py_1: py_2] = self.S21_pol_Gn[py_i, py_1: py_2] * np.sqrt(self.T[py_i]) \
                                                    * (np.sqrt(self.refractive_index_r / self.refractive_index_t));
                '# correct amplitude if top/bottom substrates are different'

            '-------------- 计算s参数 ----------------------------------'
            self.S11_x = copy.deepcopy(self.S11_Gn)
            self.S11_x = self.S11_x/np.sqrt(self.refractive_index_t/self.refractive_index_r)
            self.S21_x = copy.deepcopy(self.S21_Gn)
            self.S21_x = self.S21_x/np.sqrt(self.refractive_index_r/self.refractive_index_t)

            '保存到data'
            np.save(self.path_file_results_data + 'S21_x' + '_' + self.model_name + '.npy', self.S21_x)
            np.save(self.path_file_results_data + 'T' + '_' + self.model_name + '.npy', self.T)
            np.save(self.path_file_results_data + 'R' + '_' + self.model_name + '.npy', self.R)

            '保存到temp'
            np.save(self.path_file_temp +'S21_x.npy', self.S21_x)
            np.save(self.path_file_temp +'T.npy', self.T)
            np.save(self.path_file_temp +'R.npy', self.R)

            self.t2_calculate_S_parameters = time.time()


    '''======================================== 数据后处理 ============================================================'''
    def calculate_emw(self):
        '''calcuate right/left circular polarization components
        with a point of view from the receiver (monitor)'''
        self.t1_calculate_emw = time.time()

        '数据来自plane监视器-----------------------------------------------------------------------------------'
        self.E_r = (self.Ex - 1j * self.Ey) / np.sqrt(2);
        self.E_l = (self.Ex + 1j * self.Ey) / np.sqrt(2);
        self.H_r = (self.Hy - 1j * self.Hx) / np.sqrt(2);
        self.H_l = (self.Hy + 1j * self.Hx) / np.sqrt(2);

        '''用Python的squeeze( )函数，或者Lumerical的pinch( )函数来删除单个元素的维度'''
        self.phi_x = np.unwrap(np.angle(self.Ex)) * 180 / np.pi;
        self.phi_y = np.unwrap(np.angle(self.Ey)) * 180 / np.pi;
        self.phi_r = np.unwrap(np.angle(self.E_r)) * 180 / np.pi;
        self.phi_l = np.unwrap(np.angle(self.E_l)) * 180 / np.pi;

        '''Calculate the z component of the Poynting vector'''
        self.Pz_r = self.E_r * np.conj(self.H_l);  # 复共轭
        self.Pz_l = self.E_l * np.conj(self.H_r);

        '''Fraction of transmitted power which is RCP and LCP
        光波导基础P5公式/加了负号
        还在报错'''
        # Tr_r = -sp.integrate(Pz_r.real/ 2, 1:2, x, y) / sourcepower(f);  # 在x,y面积分
        # Tr_l = -integrate(real(Pz_l) / 2, 1:2, x, y) / sourcepower(f);

        '数据来自point监视器-----------------------------------------------------------------------------------'
        self.E_r_point = (self.Ex_point - 1j * self.Ey_point) / np.sqrt(2);
        self.E_l_point = (self.Ex_point + 1j * self.Ey_point) / np.sqrt(2);
        self.H_r_point = (self.Hy_point - 1j * self.Hx_point) / np.sqrt(2);
        self.H_l_point = (self.Hy_point + 1j * self.Hx_point) / np.sqrt(2);

        self.phi_x_point = np.unwrap(np.angle(self.Ex_point)) * 180 / np.pi;
        self.phi_y_point = np.unwrap(np.angle(self.Ey_point)) * 180 / np.pi;
        self.phi_r_point = np.unwrap(np.angle(self.E_r_point)) * 180 / np.pi;
        self.phi_l_point = np.unwrap(np.angle(self.E_l_point)) * 180 / np.pi;


        '数据来自in plane监视器-----------------------------------------------------------------------------------'
        self.E_r_in = (self.Ex_in - 1j * self.Ey_in) / np.sqrt(2);
        self.E_l_in = (self.Ex_in + 1j * self.Ey_in) / np.sqrt(2);
        self.H_r_in = (self.Hy_in - 1j * self.Hx_in) / np.sqrt(2);
        self.H_l_in = (self.Hy_in + 1j * self.Hx_in) / np.sqrt(2);
        self.phi_x_in = np.unwrap(np.angle(self.Ex_in)) * 180 / np.pi;
        self.phi_y_in = np.unwrap(np.angle(self.Ey_in)) * 180 / np.pi;
        self.phi_r_in = np.unwrap(np.angle(self.E_r_in)) * 180 / np.pi;
        self.phi_l_in = np.unwrap(np.angle(self.E_l_in)) * 180 / np.pi;

        '保存到temp'
        np.save(self.path_file_temp + 'E_r.npy', self.E_r)
        np.save(self.path_file_temp + 'E_l.npy', self.E_l)
        np.save(self.path_file_temp + 'H_r.npy', self.H_r)
        np.save(self.path_file_temp + 'H_l.npy', self.H_l)
        np.save(self.path_file_temp + 'phi_x.npy', self.phi_x)
        np.save(self.path_file_temp + 'phi_y.npy', self.phi_y)
        np.save(self.path_file_temp + 'phi_r.npy', self.phi_r)
        np.save(self.path_file_temp + 'phi_l.npy', self.phi_l)
        np.save(self.path_file_temp + 'Pz_r.npy', self.Pz_r)
        np.save(self.path_file_temp + 'Pz_l.npy', self.Pz_l)

        np.save(self.path_file_temp + 'E_r_point.npy', self.E_r_point)
        np.save(self.path_file_temp + 'E_l_point.npy', self.E_l_point)
        np.save(self.path_file_temp + 'H_r_point.npy', self.H_r_point)
        np.save(self.path_file_temp + 'H_l_point.npy', self.H_l_point)
        np.save(self.path_file_temp + 'phi_x_point.npy', self.phi_x_point)
        np.save(self.path_file_temp + 'phi_y_point.npy', self.phi_y_point)
        np.save(self.path_file_temp + 'phi_r_point.npy', self.phi_r_point)
        np.save(self.path_file_temp + 'phi_l_point.npy', self.phi_l_point)

        np.save(self.path_file_temp + 'E_r_in.npy', self.E_r_in)
        np.save(self.path_file_temp + 'E_l_in.npy', self.E_l_in)
        np.save(self.path_file_temp + 'H_r_in.npy', self.H_r_in)
        np.save(self.path_file_temp + 'H_l_in.npy', self.H_l_in)
        np.save(self.path_file_temp + 'phi_x_in.npy', self.phi_x_in)
        np.save(self.path_file_temp + 'phi_y_in.npy', self.phi_y_in)
        np.save(self.path_file_temp + 'phi_r_in.npy', self.phi_r_in)
        np.save(self.path_file_temp + 'phi_l_in.npy', self.phi_l_in)

        '保存到data'
        np.save(self.path_file_results_data + 'phi_x_point' + '_' + self.model_name + '.npy', self.phi_x_point)
        np.save(self.path_file_results_data + 'phi_y_point' + '_' + self.model_name + '.npy', self.phi_y_point)
        np.save(self.path_file_results_data + 'phi_r_point' + '_' + self.model_name + '.npy', self.phi_r_point)
        np.save(self.path_file_results_data + 'phi_l_point' + '_' + self.model_name + '.npy', self.phi_l_point)

        self.t2_calculate_emw = time.time()


    '''======================================== return 计算值 ==============================================================='''
    def get_qylumapi_phi_x_point(self):
        return self.phi_x_point
    def get_qylumapi_phi_y_point(self):
        return self.phi_y_point
    def get_qylumapi_phi_r_point(self):
        return self.phi_r_point
    def get_qylumapi_phi_l_point(self):
        return self.phi_l_point
    def get_qylumapi_S21_x(self):
        return self.S21_x
    def get_qylumapi_T(self):
        return self.T
    def get_qylumapi_R(self):
        return  self.R


    '''======================================== 可视化 ==============================================================='''
    def plot_save(self, x, y, x_label, y_label, title, fontsize, dpi, figtype):
        '比较灵活的可视化方法'
        plt.figure()
        plt.clf()
        plt.plot(x.flatten(),y.flatten())
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(self.path_file_results_figs + self.model_name + '_' + title + figtype, dpi = dpi)
        plt.close()

    def pcolor_save(self, meshgride_x, meshgride_y, data, x_label, y_label, title, fontsize, dpi, figtype):
        '比较灵活的可视化方法'
        a = len(meshgride_x); b = len(meshgride_y);
        # para = 1 + 0.1*fontsize/10 # fontsize=10差不多是刚好能显示
        plt.figure(); plt.clf();
        # plt.figure(figsize=(a * para, b * para)); plt.clf();
        '''将两个一维数组变成二维的'''
        X, Y = np.meshgrid(meshgride_x, meshgride_y)
        plt.pcolormesh(X, Y, data.T)
        plt.xlabel(x_label,fontsize=fontsize)
        plt.ylabel(y_label,fontsize=fontsize)
        plt.title(title,fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(self.path_file_results_figs + self.model_name + '_' + title + figtype, dpi = dpi)
        plt.close()

    def plot_all_save(self, fontsize, dpi, figtype):
        '把大部分可能用得到的图都绘制并保存，肯定会拖慢速度，扫描参数的时候可以不用'
        '将频率转换为波长，单位nm'
        # x_lambda = self.c / self.f * 1e9;  # [nm]
        '电磁场的数据可能对波长进行了扫描，用for循环作图'
        if self.x_lambda.size>1:
            for i in range(self.x_lambda.size):
                # for i in range(self.x_lambda.shape[0]):
                print('plot_all_save, i = ', i)
                lambda_x = np.round(self.x_lambda[i, :]);
                TPlane_Ex = self.Ex[:, :, i];
                TPlane_Ey = self.Ey[:, :, i];
                TPlane_E_r = self.E_r[:, :, i];
                TPlane_E_l = self.E_l[:, :, i];
                TPlane_phi_x = self.phi_x[:, :, i];
                TPlane_phi_y = self.phi_y[:, :, i];
                TPlane_phi_r = self.phi_r[:, :, i];
                TPlane_phi_l = self.phi_l[:, :, i];

                inPlane_Ex = self.Ex_in[:, :, i];
                inPlane_Ey = self.Ey_in[:, :, i];
                inPlane_E_r = self.E_r_in[:, :, i];
                inPlane_E_l = self.E_l_in[:, :, i];
                inPlane_phi_x = self.phi_x_in[:, :, i];
                inPlane_phi_y = self.phi_y_in[:, :, i];
                inPlane_phi_r = self.phi_r_in[:, :, i];
                inPlane_phi_l = self.phi_l_in[:, :, i];
                '可视化 T_plane monitor 振幅'
                self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_Ex.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='TPlane_Ex.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_Ey.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='TPlane_Ey.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_E_r.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='TPlane_E_r.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_E_l.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='TPlane_E_l.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                '可视化 T_plane monitor 相位'
                self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_phi_x.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='TPlane_phi_x.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_phi_y.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='TPlane_phi_y.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_phi_r.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='TPlane_phi_r.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_phi_l.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='TPlane_phi_l.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                '可视化 point monitor 相位'
                self.plot_save(x=self.x_lambda, y=self.phi_x_point
                               , x_label='wavelength (nm)', y_label='phi_x_point'
                               , title='phi_x_point.real @' + str(lambda_x) + 'nm'
                               , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.plot_save(x=self.x_lambda, y=self.phi_y_point
                               , x_label='wavelength (nm)', y_label='phi_y_point'
                               , title='phi_y_point.real @' + str(lambda_x) + 'nm'
                               , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.plot_save(x=self.x_lambda, y=self.phi_r_point
                               , x_label='wavelength (nm)', y_label='phi_r_point'
                               , title='phi_r_point.real @' + str(lambda_x) + 'nm'
                               , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.plot_save(x=self.x_lambda, y=self.phi_l_point
                               , x_label='wavelength (nm)', y_label='phi_l_point'
                               , title='phi_l_point.real @' + str(lambda_x) + 'nm'
                               , fontsize=fontsize, dpi=dpi, figtype=figtype)

                '可视化 in_plane monitor 振幅和相位'
                self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_Ex.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='inPlane_Ex.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_Ey.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='inPlane_Ey.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_E_r.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='inPlane_E_r.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_E_l.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='inPlane_E_l.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_phi_x.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='inPlane_phi_x.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_phi_y.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='inPlane_phi_y.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_phi_r.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='inPlane_phi_r.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
                self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_phi_l.real
                                 , x_label='x-direction (nm)', y_label='y-direction (nm)',
                                 title='inPlane_phi_l.real @' + str(lambda_x) + 'nm'
                                 , fontsize=fontsize, dpi=dpi, figtype=figtype)
        else:
            TPlane_Ex = self.Ex;
            TPlane_Ey = self.Ey;
            TPlane_E_r = self.E_r;
            TPlane_E_l = self.E_l;
            TPlane_phi_x = self.phi_x;
            TPlane_phi_y = self.phi_y;
            TPlane_phi_r = self.phi_r;
            TPlane_phi_l = self.phi_l;
            lambda_x = np.round(self.x_lambda);

            inPlane_Ex = self.Ex_in;
            inPlane_Ey = self.Ey_in;
            inPlane_E_r = self.E_r_in;
            inPlane_E_l = self.E_l_in;
            inPlane_phi_x = self.phi_x_in;
            inPlane_phi_y = self.phi_y_in;
            inPlane_phi_r = self.phi_r_in;
            inPlane_phi_l = self.phi_l_in;
            '可视化 T_plane monitor 振幅'
            self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_Ex.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='TPlane_Ex.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_Ey.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='TPlane_Ey.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_E_r.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='TPlane_E_r.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_E_l.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='TPlane_E_l.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            '可视化 T_plane monitor 相位'
            self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_phi_x.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='TPlane_phi_x.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_phi_y.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='TPlane_phi_y.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_phi_r.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='TPlane_phi_r.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x * 1e9, meshgride_y=self.y * 1e9, data=TPlane_phi_l.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='TPlane_phi_l.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            '可视化 point monitor 相位'
            self.plot_save(x=self.x_lambda, y=self.phi_x_point
                           , x_label='wavelength (nm)', y_label='phi_x_point'
                           , title='phi_x_point.real @' + str(lambda_x) + 'nm'
                           , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.plot_save(x=self.x_lambda, y=self.phi_y_point
                           , x_label='wavelength (nm)', y_label='phi_y_point'
                           , title='phi_y_point.real @' + str(lambda_x) + 'nm'
                           , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.plot_save(x=self.x_lambda, y=self.phi_r_point
                           , x_label='wavelength (nm)', y_label='phi_r_point'
                           , title='phi_r_point.real @' + str(lambda_x) + 'nm'
                           , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.plot_save(x=self.x_lambda, y=self.phi_l_point
                           , x_label='wavelength (nm)', y_label='phi_l_point'
                           , title='phi_l_point.real @' + str(lambda_x) + 'nm'
                           , fontsize=fontsize, dpi=dpi, figtype=figtype)

            '可视化 point monitor 相位, 就一个lambda，没法画'

            '可视化 in_plane monitor 振幅和相位'
            self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_Ex.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='inPlane_Ex.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_Ey.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='inPlane_Ey.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_E_r.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='inPlane_E_r.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_E_l.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='inPlane_E_l.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_phi_x.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='inPlane_phi_x.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_phi_y.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='inPlane_phi_y.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_phi_r.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='inPlane_phi_r.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)
            self.pcolor_save(meshgride_x=self.x_in * 1e9, meshgride_y=self.y_in * 1e9, data=inPlane_phi_l.real
                             , x_label='x-direction (nm)', y_label='y-direction (nm)',
                             title='inPlane_phi_l.real @' + str(lambda_x) + 'nm'
                             , fontsize=fontsize, dpi=dpi, figtype=figtype)


    def show_info(self):
        print('\n============================= qulumapi 参数设置 ======================================== ')
        print('基底：厚度 sub_d = ', self.sub_d, ', sub_z_min=', self.sub_z_min)
        print('基底：长度 Ux = ', self.Ux, ', 宽度 Uy = ', self.Uy)
        print('meta/film：厚度 film_d = ', self.film_d)
        print('超表面单元：element_size_L = ', self.element_size_L, ' element_size_W = ', self.element_size_W)
        print('自定义网格：范围 z_meta_max = ', self.z_meta_max)
        print('自定义网格：中心 z_meta_min = ', self.z_meta_min)

        print('仿真区域：范围 z_fdtd_max = ', self.z_fdtd_max)
        print('仿真区域：中心 z_fdtd_min = ', self.z_fdtd_min)
        print('自定义网格：范围 z_meta_max = ', self.z_meta_max)
        print('自定义网格：中心 z_meta_min = ', self.z_meta_min)

        print("光源：频率点 self.nf = ", self.nf, '; self.freq_num =', self.freq_num, '; lambda = ', self.x_lambda)
        print('光源：位置 source_z = ', self.source_z)
        print('反射面：平面  monitor_plane_R_z = ', self.monitor_plane_R_z)
        print('反射面：点    monitor_point_R_z = ', self.monitor_point_R_z)
        print('反射面：index monitor_index_R_z = ', self.monitor_index_R_z)
        print('入射面：平面  monitor_plane_in_z = ', self.monitor_plane_in_z)
        print('入射面：点    monitor_point_in_z = ', self.monitor_point_in_z)
        print('入射面：index monitor_index_in_z = ', self.monitor_index_in_z)
        print('透射面：平面  monitor_plane_T_z = ', self.monitor_plane_T_z)
        print('透射面：点    monitor_point_T_z = ', self.monitor_point_T_z)
        print('透射面：index monitor_index_T_z = ', self.monitor_index_T_z)
        print('-------------- 计算结果 ----------------------------------')
        print('self.S11_Gn = ', type(self.S11_Gn), self.S11_Gn.shape, self.S11_Gn)
        print('self.S21_Gn = ', type(self.S21_Gn), self.S21_Gn.shape, self.S21_Gn)
        print('self.S11_pol_Gn = ', type(self.S11_pol_Gn), self.S11_pol_Gn.shape, self.S11_pol_Gn)
        print('self.S21_pol_Gn = ', type(self.S21_pol_Gn), self.S21_pol_Gn.shape, self.S21_pol_Gn)
        print('self.T = ', type(self.T), self.T)
        print('self.R = ', type(self.R), self.R)
        print('self.phi_r_point = ', type(self.phi_r_point), self.phi_r_point)
        print('self.phi_l_point = ', type(self.phi_l_point), self.phi_l_point)
        print('对于平面结构定义的S参数中相关偏振参数的约定见下图。如果超材料上下方的介质具有不同的折射率，|S12|^2 和|S21|^2 都大于1')


'================================================= main ==============================================================='
def main():
    '---------- 路径 ----------'
    print('path_file = ',para_str['path_file'])

    '---------- 建模与仿真计算 ----------'
    print('正在调用FDTD建模与仿真计算')
    model_name = para_str['model_name'] + '_' + para_source['plane_wave_type'][:5]
    Class1 = QYFDTD(path_file=para_str['path_file'], model_name=model_name
                    , lambda_start=para_source['lambda_start']
                    , lambda_stop=para_source['lambda_stop']
                    , freq_num=para_source['freq_num']
                    , element_rotation_z=para_frame['rotation_z_element']
                    , element_size_1=para_frame['element_d']
                    , element_size_2=para_element['element_rect_L']
                    , element_size_3=para_element['element_rect_W']
                    , element_size_4=0
                    )


    '---------- 建立模型 ----------'
    Class1.build_structure()

    '---------- 数据后处理 ----------'
    print('正在提取、计算、保存数据')
    Class1.run_model()
    # Class1.save_fdtd_raw_data()
    # Class1.calculate_emw()
    # print('正在计算 S参数')
    # Class1.calculate_S_parameters()

    # '---------- 可视化 ----------'
    # print('正在绘制、保存图片')
    # Class1.plot_all_save(fontsize=para_fig1['fontsize'], dpi=para_fig1['dpi'], figtype=para_fig1['figtype'])
    # Class1.show_info()


'================================================= 调试 ================================================================'
if __name__ == '__main__':
    # “Make a script both importable and executable”意思就是说让你写的脚本模块既可以导入到别的模块中用，另外该模块自己也可执行。
    # "这部分代码只有在调试的时候才会显示，被调用时不显示"
    print("\n%s is being run by itself"%__name__)
    t1 = time.time()
    main()
    print("\n%s is run Over" % __name__)

    print("---------- 调试内容 ----------")

    t2 = time.time()
    print('t2-t1 = ', (t2 - t1),'[s] = ', (t2 - t1)/60/60,'[h]')
    print("---------- 调试结束 ----------")

else:
    # "这部分代码只有在调用的时候才会显示，调试时不显示"
    print("\n%s is being imported from another module" % __name__)


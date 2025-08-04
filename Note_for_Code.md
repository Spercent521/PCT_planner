原始文件名 : Note_for_Code.md

>   [!IMPORTANT]
>
>   **Requirements :**
>
>   - Ubuntu >= 20.04
>   - ROS >= Noetic with ros-desktop-full installation
>   - CUDA >= 11.7
>
>   - Python >= 3.8
>   - CuPy with CUDA >= 11.7
>   - Open3d
>
>   ---
>
>   `3rdparty/gtsam-4.1.1`
>
>   `osqp`

# Tomography

```bash
cd tomography/scripts/
python3 tomography.py --scene Spiral
```

```python
# tomography.py

if __name__ == '__main__':
    '''
    用 argparse 模块解析参数 --scene " Spiral , Building , Plaza "
    '''
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
    args = parser.parse_args()

    '''
    加载全局默认配置 如 ROS话题名、导出路径
    '''
    cfg = Config()

    '''
    根据 --scene 参数动态导入对应的 类 处理数据 
    '''
    scene_cfg = getattr(__import__('config'), 'Scene' + args.scene)

    '''
    ROS 节点初始化 
    节点名称 : pointcloud_tomography
    anonymous=True : 自动追加随机后缀 避免多实例冲突
    '''
    rospy.init_node('pointcloud_tomography', anonymous=True)

    '''
    启动处理模块 转到 Tomography.__init__()
    '''
    mapping = Tomography(cfg, scene_cfg)

    '''
    阻止Python程序退出 保持节点运行
    - 维持ROS话题发布
    - 响应可能的服务调用
    - 监听退出信号

    若需周期性执行 , 可用 `rospy.Timer` 替代
    '''
    rospy.spin()
```

>   [!NOTE]
>
>   1.   加载全局配置
>   2.   根据 --scene 参数导入对应的 类 处理初始点云数据
>   3.   `mapping = Tomography(cfg, scene_cfg)`启动处理模块 , 转到 `Tomography.__init__()`

```python
# tomography.py

class Tomography(object):
    def __init__(self, cfg, scene_cfg):
        self.export_dir = rsg_root + cfg.map.export_dir # 导出目录完整路径
        self.pcd_file = scene_cfg.pcd.file_name         # 点云文件名（如"Spiral.pcd"）
        self.resolution = scene_cfg.map.resolution      # 地图网格分辨率（单位：米/格）
        self.ground_h = scene_cfg.map.ground_h          # 地面基准高度
        self.slice_dh = scene_cfg.map.slice_dh          # 切片高度间隔

        self.center = np.zeros(2, dtype=np.float32)     # 地图中心坐标（初始化为[0,0]）
        self.tomogram = Tomogram(scene_cfg)             # 创建Tomogram处理实例
        points = self.loadPCD(self.pcd_file)

        # Process
        self.process(points)
```

>   [!NOTE]
>
>   初始化之后转到 `self.process(points)`
>
>   Class Tomography 定义的函数如下 :
>
>   ![demo](Note_for_Code_Pic/class Tomography(object).png)

```python
# tomography.py
# def process(self, points)

def process(self, points):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 10

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            # point2map
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, t_gpu = self.tomogram.point2map(points)

            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        rospy.loginfo("Num slices simp: %d", layers_g.shape[0])
        rospy.loginfo("Num repeats (for benchmarking only): %d", n_repeat)
        rospy.loginfo(" -- avg t_map  (ms): %f", t_map / n_repeat)
        rospy.loginfo(" -- avg t_trav (ms): %f", t_trav / n_repeat)
        rospy.loginfo(" -- avg t_simp (ms): %f", t_simp / n_repeat)
        rospy.loginfo(" -- avg t_all  (ms): %f", t_all / n_repeat)

        # 数据导出与发布
        self.n_slice = layers_g.shape[0]

        map_file = os.path.splitext(self.pcd_file)[0]
        self.exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file)

        self.initROS()
        self.publishPoints(points)
        self.publishLayers(self.layer_G_pub_list, layers_g, layers_t)
        self.publishLayers(self.layer_C_pub_list, layers_c, None)
        self.publishTomogram(layers_g, layers_t)
```

>   [!NOTE]
>
>   核心就是 : `[Line 20] self.tomogram.point2map(points)`
>
>   后面主要就是发布Topic

## Point2map

```python
# tomogram.py
class Tomogram(object):
	"""
	其他函数...
	"""
    def point2map(self, points):
            """
            # 数据预处理 包括 CuPy 实现 零拷贝传输
            
            III. METHODOLOGY
            A. Tomogram Construction
            B. Traversability Estimation
            C. Tomogram Simplification
            
            # RETURN
      		都是 测评参数 
      		"""
```

### III. METHODOLOGY A. Tomogram Construction

```python
# tomogram.py
    def point2map(self, points):
    	# ...
        self.tomography_kernel(
            '''
            按照高度切片 计算e^{g,c}
            '''
            points, self.center, 
            self.layers_g, self.layers_c,
            size=(points.shape[0])
        )

        diff_x_sq = cp.maximum(
            # x 方向上的差分平方
            (self.layers_g[:, 1:-1, :] - self.layers_g[:, :-2, :]) ** 2, 
            (self.layers_g[:, 1:-1, :] - self.layers_g[:,  2:, :]) ** 2
        )
        diff_y_sq = cp.maximum(
            # y 方向上的差分平方
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :, :-2]) ** 2, 
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :,  2:]) ** 2
        )
        self.grad_mag_sq[:, 1:-1, 1:-1] = diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
        self.grad_mag_max[:, 1:-1, 1:-1] = cp.maximum(diff_x_sq[:, :, 1:-1], diff_y_sq[:, 1:-1, :])
        
        # 求间隔 ceiling - ground
        interval = (self.layers_c - self.layers_g)
        # ...
```

```python
# kernels.py
'''
self.tomography_kernel(
            points, self.center, 
            self.layers_g, self.layers_c,
            size=(points.shape[0])
        )
'''
def tomographyKernel(resolution, n_row, n_col, n_slice, slice_h0, slice_dh):
    tomography_kernel = cp.ElementwiseKernel(
        in_params='raw U points, raw U center',
        out_params='raw U layers_g, raw U layers_c',
        preamble=utils_point(resolution, n_row, n_col),
        operation=string.Template(
            '''
            U px = points[i * 3];
            U py = points[i * 3 + 1];
            U pz = points[i * 3 + 2];

            int idx = getIndexMap_1d(px, py, center[0], center[1]);
            if ( idx < 0 ) 
                return; 
            for ( int s_idx = 0; s_idx < ${n_slice}; s_idx ++ )
            {
                U slice = ${slice_h0} + s_idx * ${slice_dh};
                if ( pz <= slice )
                    atomicMaxFloat(&layers_g[getIndexBlock_1d(idx, s_idx)], pz);
                else
                    atomicMinFloat(&layers_c[getIndexBlock_1d(idx, s_idx)], pz);
            }
            '''
        ).substitute(
            n_slice=n_slice,
            slice_h0=slice_h0,
            slice_dh=slice_dh
        ),
        name='tomography_kernel'
    )
                            
    return tomography_kernel
```

### III. METHODOLOGY B. Traversability Estimation

```python
# tomogram.py	
	def point2map(self, points):
    	# ...	
    	self.trav_kernel(
            # travKernel function in kernels.py
            interval, self.grad_mag_sq, self.grad_mag_max,
            self.trav_cost,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )

        self.inflation_kernel(
            self.trav_cost, self.inf_table,
            self.inflated_cost,
            size=(self.n_slice_init * self.map_dim_x * self.map_dim_y)
        )
        # ...	
```

```python
# kernels.py
'''
self.trav_kernel = travKernel(
            self.map_dim_x,
            self.map_dim_y,
            self.half_trav_k_size,
            self.interval_min,
            self.interval_free,
            self.step_cross, 
            self.step_stand, 
            self.standable_th, 
            self.cost_barrier
        )
'''
def travKernel(
    n_row, n_col, half_kernel_size,
    interval_min, interval_free, step_cross, step_stand, standable_th, cost_barrier
    ):
    trav_kernel = cp.ElementwiseKernel(
        in_params='raw U interval, raw U grad_mag_sq, raw U grad_mag_max',
        out_params='raw U trav_cost',
        preamble=utils_map(n_row, n_col),
        operation=string.Template(
            '''
            if ( interval[i] < ${interval_min} )
            {
                trav_cost[i] = ${cost_barrier};
                return;
            }
            else
                trav_cost[i] += max(0.0, 20 * (${interval_free} - interval[i]));
            if ( grad_mag_sq[i] <= ${step_stand_sq} )
            {
                trav_cost[i] += 15 * grad_mag_sq[i] / ${step_stand_sq};
                return;
            }
            else 
            {
                if ( grad_mag_max[i] <= ${step_cross_sq} )
                {
                    int standable_grids = 0;
                    for ( int dy = -${half_kernel_size}; dy <= ${half_kernel_size}; dy++ ) 
                    {
                        for ( int dx = -${half_kernel_size}; dx <= ${half_kernel_size}; dx++ ) 
                        {
                            int idx = getIdxRelative(i, dx, dy);
                            if ( idx < 0 )
                                continue;
                            if ( grad_mag_sq[idx] < ${step_stand_sq} )
                                standable_grids += 1;
                        }
                    }
                    if ( standable_grids < ${standable_th} )
                    {
                        trav_cost[i] = ${cost_barrier};
                        return;
                    }
                    else
                        trav_cost[i] += 20 * grad_mag_max[i] / ${step_cross_sq};
                }
                else
                {
                    trav_cost[i] = ${cost_barrier};
                    return;
                }
            } 
            '''
        ).substitute(
            half_kernel_size=half_kernel_size,
            interval_min=interval_min,
            interval_free=interval_free,
            step_cross_sq=step_cross ** 2,
            step_stand_sq=step_stand ** 2,
            standable_th=standable_th,
            cost_barrier=cost_barrier
        ),
        name='trav_kernel'
    )
                            
    return trav_kernel
```

>   [!NOTE]
>
>   ####  **垂直空间约束**
>
>   -   若垂直空间 `< interval_min`（机器人最小通过高度），代价直接设为最大值 `cost_barrier`（不可通行）。
>   -   否则，代价随空间减少线性增加（`interval_free` 是理想通过高度）。
>
>   #### **坡度评估**
>
>   需要根据我们自己的机器人设计 比如他们的四足机器人能通过一些我们轮式机器人不可通过的点
>
>   #### **总之**
>
>   会生成代价地图 不可通信设置为 $+ \infty$

```python
# kernels.py
'''
self.inflation_kernel = inflationKernel(
    self.map_dim_x,
    self.map_dim_y,
    self.half_inf_k_size
)
'''
def inflationKernel(n_row, n_col, half_kernel_size):
    inflation_kernel = cp.ElementwiseKernel(
        in_params='raw U trav_cost, raw U score_table',
        out_params='raw U inflated_cost',
        preamble=utils_map(n_row, n_col),
        operation=string.Template(
            '''
            int counter = 0;
            for ( int dy = -${half_kernel_size}; dy <= ${half_kernel_size}; dy++ ) 
            {
                for ( int dx = -${half_kernel_size}; dx <= ${half_kernel_size}; dx++ ) 
                {
                    int idx = getIdxRelative(i, dx, dy);
                    if ( idx >= 0 )
                        inflated_cost[i] = max(inflated_cost[i], trav_cost[idx] * score_table[counter]);
                    counter += 1;
                }
            }
            '''
        ).substitute(
            half_kernel_size=half_kernel_size
        ),
        name='inflation_kernel'
    )
                            
    return inflation_kernel
```

>   [!NOTE]
>
>   膨胀处理 在障碍物周围膨胀出更高的代价

### III. METHODOLOGY C. Tomogram Simplification

```python
# tomogram.py
    def point2map(self, points):
    	# ...
		idx_simp = [0]											# 存储的是 最终保留的层索引
																# 初始化为 [0]，即默认包含第 0 层
        if self.layers_g.shape[0] > 1:
            l_idx, m_idx = 0, 1									# l_idx 上一层 , m_idx 本层
            diff_h = self.layers_g[1:] - self.layers_g[:-1]
            while m_idx < self.n_slice_init - 2:				# 循环变量 m_idx = layer_number
                mask_l_g = self.layers_g[m_idx] - self.layers_g[l_idx] > 0
                mask_l_t = self.inflated_cost[l_idx] > self.inflated_cost[m_idx]
                mask_u_g = diff_h[m_idx] > 0
                mask_t = self.inflated_cost[m_idx] < self.cost_barrier
                unique = (mask_l_g | mask_l_t) & mask_u_g & mask_t			# def unique 
                if cp.any(unique):											# cupy 实现 逻辑或
                    idx_simp.append(m_idx)
                    l_idx = m_idx
                m_idx += 1
            idx_simp.append(m_idx)

        trav_grad_x = (self.inflated_cost[idx_simp][:, 2:, :] - self.inflated_cost[idx_simp][:, :-2, :])
        trav_grad_y = (self.inflated_cost[idx_simp][:, :, 2:] - self.inflated_cost[idx_simp][:, :, :-2])
        # ...
```

>   [!NOTE]
>
>   -   **`(mask_l_g | mask_l_t)`**：当前层比上次保留的层 **更高** 或 **成本更低**。
>   -   **`& mask_u_g`**：当前层比下一层 **更高**（防止连续下降的层被保留）。
>   -   **`& mask_t`**：当前层成本 **低于障碍阈值**（确保不是障碍物）。

# D. Path Planning through Slices

```bash
cd planner/scripts/
python3 plan.py --scene Spiral
```


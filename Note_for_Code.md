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
	def point2map(self, points):
    	# ...
        self.tomography_kernel(
            points, self.center, 
            self.layers_g, self.layers_c,
            size=(points.shape[0])
        )

        diff_x_sq = cp.maximum(
            (self.layers_g[:, 1:-1, :] - self.layers_g[:, :-2, :]) ** 2, 
            (self.layers_g[:, 1:-1, :] - self.layers_g[:,  2:, :]) ** 2
        )
        diff_y_sq = cp.maximum(
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :, :-2]) ** 2, 
            (self.layers_g[:, :, 1:-1] - self.layers_g[:, :,  2:]) ** 2
        )
        self.grad_mag_sq[:, 1:-1, 1:-1] = diff_x_sq[:, :, 1:-1] + diff_y_sq[:, 1:-1, :]
        self.grad_mag_max[:, 1:-1, 1:-1] = cp.maximum(diff_x_sq[:, :, 1:-1], diff_y_sq[:, 1:-1, :])
        
        interval = (self.layers_c - self.layers_g)
        # ...
```



### III. METHODOLOGY B. Traversability Estimation

```python
	def point2map(self, points):
    	# ...	
    	self.trav_kernel(
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



### III. METHODOLOGY C. Tomogram Simplification

```python
	def point2map(self, points):
    	# ...
		idx_simp = [0]
        if self.layers_g.shape[0] > 1:
            l_idx, m_idx = 0, 1
            diff_h = self.layers_g[1:] - self.layers_g[:-1]
            while m_idx < self.n_slice_init - 2:
                mask_l_g = self.layers_g[m_idx] - self.layers_g[l_idx] > 0
                mask_l_t = self.inflated_cost[l_idx] > self.inflated_cost[m_idx]
                mask_u_g = diff_h[m_idx] > 0
                mask_t = self.inflated_cost[m_idx] < self.cost_barrier
                unique = (mask_l_g | mask_l_t) & mask_u_g & mask_t
                if cp.any(unique):
                    idx_simp.append(m_idx)
                    l_idx = m_idx
                m_idx += 1
            idx_simp.append(m_idx)

        trav_grad_x = (self.inflated_cost[idx_simp][:, 2:, :] - self.inflated_cost[idx_simp][:, :-2, :])
        trav_grad_y = (self.inflated_cost[idx_simp][:, :, 2:] - self.inflated_cost[idx_simp][:, :, :-2])
        # ...
```


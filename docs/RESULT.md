<img src="../docs/figs/logo.png" align="right" width="20%">

# Experimental Result

- [Track 1: Uniform Split](#track-1-uniform-split)
   - [Range View](#range-view)
   - [Voxel](#voxel)
- [Track 2: Sequential Split](#track-2-sequential-split)
   - [Range View](#range-view-1)
   - [Voxel](#voxel-1) 

## Track 1: Uniform Split
> This track is analogous to the semi-supervised image segmentation community, which sample LiDAR scans with a uniform probability.

### Range View

<table>
   <tr>
      <th rowspan="2">Method</th>
      <th colspan="5">nuScenes</th>
      <th colspan="5">SemanticKITTI</th>
      <th colspan="5">ScribbleKITTI</th>
   </tr>
   <tr>
      <td>1%</td> <td>10%</td> <td>20%</td> <td>50%</td> <td>Full</td>
      <td>1%</td> <td>10%</td> <td>20%</td> <td>50%</td> <td>Full</td>
      <td>1%</td> <td>10%</td> <td>20%</td> <td>50%</td> <td>Full</td>
   </tr>
   
   
   <tr>
      <td><a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf">RangeNet++</a></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><strong><i>w/</i> LaserMix</strong></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><i>improv.</i> &#8593</td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
   </tr>
   <tr>
      <td>Download</td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
   </tr>
   <tr>
      <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   
   
   <tr>
      <td><a href="https://arxiv.org/abs/2003.03653">SalsaNext</a></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><strong><i>w/</i> LaserMix</strong></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><i>improv.</i> &#8593</td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
   </tr>
   <tr>
      <td>Download</td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
   </tr>
   <tr>
      <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   
   
   <tr>
      <td><a href="https://arxiv.org/abs/2109.03787">FIDNet</a></td>
      <td>38.3</td> <td>57.5</td> <td>62.7</td> <td>67.6</td> <td>72.1</td>
      <td>36.2</td> <td>52.2</td> <td>55.9</td> <td>57.2</td> <td>61.4</td>
      <td>33.1</td> <td>47.7</td> <td>49.9</td> <td>52.5</td> <td> </td>
   </tr>
   <tr>
      <td><strong><i>w/</i> LaserMix</strong></td>
      <td>49.5</td> <td>68.2</td> <td>70.6</td> <td>73.0</td> <td> </td>
      <td>43.4</td> <td>58.8</td> <td>59.4</td> <td>61.4</td> <td> </td>
      <td>38.3</td> <td>54.4</td> <td>55.6</td> <td>58.7</td> <td> </td>
   </tr>
   <tr>
      <td><i>improv.</i> &#8593</td>
      <td><sup>+</sup>11.2</td> <td><sup>+</sup>10.7</td> <td><sup>+</sup>7.9</td> <td><sup>+</sup>5.4</td> <td><sup>+</sup></td>
      <td><sup>+</sup>7.2</td> <td><sup>+</sup>6.6</td> <td><sup>+</sup>3.5</td> <td><sup>+</sup>4.2</td> <td><sup>+</sup></td>
      <td><sup>+</sup>5.2</td> <td><sup>+</sup>6.7</td> <td><sup>+</sup>5.7</td> <td><sup>+</sup>6.2</td> <td><sup>+</sup></td>
   </tr>
   <tr>
      <td>Download</td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
   </tr>
   <tr>
      <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   
   
   <tr>
      <td><a href="https://arxiv.org/abs/2207.12691">CENet</a></td>
      <td>40.4</td> <td>58.7</td> <td> </td> <td> </td> <td>71.7</td>
      <td>40.4</td> <td>56.0</td> <td>58.2</td> <td>60.1</td> <td>62.9</td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><strong><i>w/</i> LaserMix</strong></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><i>improv.</i> &#8593</td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
   </tr>
   <tr>
      <td>Download</td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
   </tr>
   <tr>
      <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   
</table>



### Voxel

<table>
   <tr>
      <th rowspan="2">Method</th>
      <th colspan="5">nuScenes</th>
      <th colspan="5">SemanticKITTI</th>
      <th colspan="5">ScribbleKITTI</th>
   </tr>
   <tr>
      <td>1%</td> <td>10%</td> <td>20%</td> <td>50%</td> <td>Full</td>
      <td>1%</td> <td>10%</td> <td>20%</td> <td>50%</td> <td>Full</td>
      <td>1%</td> <td>10%</td> <td>20%</td> <td>50%</td> <td>Full</td>
   </tr>
   
   
   <tr>
      <td><a href="https://github.com/NVIDIA/MinkowskiEngine">MinkUNet</a></td>
      <td> </td> <td>69.4</td> <td> </td> <td> </td> <td>72.8</td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td>62.8</td>
      <td></td> <td></td> <td></td> <td></td> <td> </td>
   </tr>
   <tr>
      <td><strong><i>w/</i> LaserMix</strong></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><i>improv.</i> &#8593</td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
   </tr>
   <tr>
      <td>Download</td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
   </tr>
   <tr>
      <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   
   
   <tr>
      <td><a href="https://arxiv.org/pdf/2007.16100">SPVCNN</a></td>
      <td>53.7</td> <td>69.7</td> <td>71.6</td> <td>72.0</td> <td>73.2</td>
      <td>42.7</td> <td>57.5</td> <td>60.2</td> <td>62.7</td> <td>63.2</td>
      <td>38.2</td> <td>53.8</td> <td> </td> <td> </td> <td>56.7</td>
   </tr>
   <tr>
      <td><strong><i>w/</i> LaserMix</strong></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td>65.8</td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><i>improv.</i> &#8593</td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup>2.6</td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
   </tr>
   <tr>
      <td>Download</td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
   </tr>
   <tr>
      <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   
   
   <tr>
      <td><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Cylindrical_and_Asymmetrical_3D_Convolution_Networks_for_LiDAR_Segmentation_CVPR_2021_paper.pdf">Cylinder3D</a></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><strong><i>w/</i> LaserMix</strong></td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
   <tr>
      <td><i>improv.</i> &#8593</td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
      <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td> <td><sup>+</sup></td>
   </tr>
   <tr>
      <td>Download</td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
      <td> </td> <td><a href="">[link]</a></td> <td> </td> <td> </td> <td><a href="">[link]</a></td>
   </tr>
   <tr>
      <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
      <td> </td> <td> </td> <td> </td> <td> </td> <td> </td>
   </tr>
</table>




## Track 2: Sequential Split
> This track takes into account the LiDAR data collection nature when sampling LiDAR scans.

### Range View
Available soon.

### Voxel 
Available soon.



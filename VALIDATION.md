# Placement and grasp validation

The contact regression used seed 497922811. Before the height adjustment,
both fingers contacted the source table and neither contacted blue at closure.
Raising the grasp and release reference by 8 mm cleared the table. Both fingers
contacted blue after closure and lift, and all three blocks transferred.

The following fresh layouts used red (80, 80), blue (185, 145), and green
(280, 75) in destination-table millimetres. Values are settled radial errors.

| Seed | Red (mm) | Blue (mm) | Green (mm) | Verified transfers |
|---|---:|---:|---:|---:|
| 497922812 | 12.4 | 8.7 | 7.7 | 3/3 |
| 497922813 | 8.9 | 10.8 | 6.6 | 3/3 |
| 497922814 | 11.0 | 6.3 | 5.0 | 3/3 |

Mean radial error was 8.6 mm. Along X, the gripper averaged 3.8 mm beyond
the requested coordinate, and the held block centre averaged a further 4.8 mm
beyond the gripper. From opening to settling, mean planar movement was 0.4 mm.
These trials used in-process messages and a software-rendered mapping image.
Live ROS mapping was separately observed. The corrected grasp has been verified
in headless physics. Reliability across hardware and unseen operating conditions
remains unmeasured.

[Raw trial records, including the original failure](evidence/grasp_height_trials/)

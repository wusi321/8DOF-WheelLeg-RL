from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformThresholdVelocityCommand(UniformVelocityCommand):
    """甯︽鍖鸿繃婊ゃ€侀珮閫熶晶鍚戣В鑰︿互鍙婂湴褰㈣嚜閫傚簲閫熷害绾︽潫鐨勯€熷害鍛戒护鐢熸垚鍣ㄣ€?
    
    鍦ㄦゼ姊拰鍨傜洿鐭绛夐殰纰嶅湴褰笂锛岃嚜鍔ㄥ皢 y 杞达紙妯Щ锛夊拰 z 杞达紙鍋忚埅锛夋帶鍒剁害鏉熺疆 0.0锛?
    寮哄埗鏈哄櫒浜哄績鏃犳梺楠涚瑪鐩村啿閿嬭秺闅滐紝鏈夋晥鏉滅粷鎵撴粦鍜屽亸鑸炕婊氳穼钀斤紱鑰屽湪骞冲湴銆佹枩鍧＄瓑鏅€氬湴褰笂
    鏀惧紑澶氬悜娣峰悎閲囨牱锛岃缁冩満鍔ㄨ浆鍚戣兘鍔涖€?
    """
    cfg: UniformThresholdVelocityCommandCfg

    def __init__(self, cfg: UniformThresholdVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.is_lateral_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_yaw_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.was_climbing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 缂撳瓨鍦板舰绫诲瀷鐨勭储寮曪紙鍙伴樁銆佸弽鍚戝彴闃躲€佸瀭鐩寸煭澧欙級锛屽疄鐜伴珮瀹归敊鍔ㄦ€佹煡鎵?
        self._climbing_indices = []
        self._flat_index = -1
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is not None and getattr(terrain.cfg, "terrain_generator", None) is not None:
            sub_terrain_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
            for name in ["pyramid_stairs", "pyramid_stairs_inv", "rc_wall"]:
                if name in sub_terrain_names:
                    self._climbing_indices.append(sub_terrain_names.index(name))
            if "flat" in sub_terrain_names:
                self._flat_index = sub_terrain_names.index("flat")

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # 1. 璋冪敤鍩虹被鐨勬爣鍑嗛噰鏍?
        super()._resample_command(env_ids)
        
        # 2. 鍩虹姝诲尯杩囨护锛氫綆浜?0.2 m/s 鏃跺己鍒跺綊闆讹紝鍖哄垎闈欐鍜岃繍鍔?
        cmd_xy_norm = torch.norm(self.vel_command_b[env_ids, :2], dim=1)
        small_cmd_mask = cmd_xy_norm < 0.2
        small_cmd_ids = env_ids[small_cmd_mask]
        if len(small_cmd_ids) > 0:
            self.vel_command_b[small_cmd_ids, :] = 0.0
            self.vel_command_w[small_cmd_ids, :] = 0.0

        # 閲嶇疆骞堕噰鏍峰崟 Y锛堝彧妯Щ锛変笌鍗?Z锛堝彧鍘熷湴杞悜锛夋寚浠ゅ垎甯?
        self.is_lateral_env[env_ids] = False
        self.is_yaw_env[env_ids] = False

        # 瀵规病鏈夎涓哄墠杩涗笖娌℃湁闈欐鐨勬縺娲荤幆澧冭繘琛屽崟杞撮噰鏍?
        active_non_fwd_mask = (~self.is_forward_env[env_ids]) & (~self.is_standing_env[env_ids])
        active_non_fwd_ids = env_ids[active_non_fwd_mask]

        if len(active_non_fwd_ids) > 0:
            r = torch.empty(len(active_non_fwd_ids), device=self.device)
            # 閲囨牱鍗?Y 鍗犳瘮
            self.is_lateral_env[active_non_fwd_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_lateral_envs
            lat_ids = active_non_fwd_ids[self.is_lateral_env[active_non_fwd_ids]]
            if len(lat_ids) > 0:
                self.vel_command_b[lat_ids, 0] = 0.0
                y_signs = self.vel_command_b[lat_ids, 1].sign()
                y_signs[y_signs == 0] = 1.0
                self.vel_command_b[lat_ids, 1] = y_signs * self.vel_command_b[lat_ids, 1].abs().clamp(min=0.3)
                self.vel_command_b[lat_ids, 2] = 0.0

            # 瀵逛笉鏄?forward 涔熶笉鏄?lateral 鐨勭幆澧冮噰鏍峰崟 Z 鍗犳瘮
            non_lat_mask = ~self.is_lateral_env[active_non_fwd_ids]
            non_lat_ids = active_non_fwd_ids[non_lat_mask]

            if len(non_lat_ids) > 0:
                r_yaw = torch.empty(len(non_lat_ids), device=self.device)
                self.is_yaw_env[non_lat_ids] = r_yaw.uniform_(0.0, 1.0) <= self.cfg.rel_yaw_envs
                yaw_ids = non_lat_ids[self.is_yaw_env[non_lat_ids]]
                if len(yaw_ids) > 0:
                    self.vel_command_b[yaw_ids, 0] = 0.0
                    self.vel_command_b[yaw_ids, 1] = 0.0
                    z_signs = self.vel_command_b[yaw_ids, 2].sign()
                    z_signs[z_signs == 0] = 1.0
                    self.vel_command_b[yaw_ids, 2] = z_signs * self.vel_command_b[yaw_ids, 2].abs().clamp(min=0.3)

        # 3. 鍦板舰鑷€傚簲閲嶉噰鏍烽檺鍒讹細鑻ュ湪鐖鍦板舰锛屽己鍒剁函鍓嶈繘鏂瑰悜涓旈€熷害 >= 0.3 m/s
        terrain = getattr(self._env.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is not None and len(self._climbing_indices) > 0:
            is_climbing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for idx in self._climbing_indices:
                is_climbing |= (terrain_types == idx)
                
            climbing_env_ids = env_ids[is_climbing[env_ids]]
            if len(climbing_env_ids) > 0:
                # 寮哄埗 x 鏂瑰悜涓烘锛堝悜鍓嶈秺闅滐級锛寉 鍜?z 杞存寚浠ゆ竻闆?
                self.vel_command_b[climbing_env_ids, 0] = self.vel_command_b[climbing_env_ids, 0].abs().clamp(min=0.3)
                self.vel_command_b[climbing_env_ids, 1] = 0.0
                self.vel_command_b[climbing_env_ids, 2] = 0.0
                self.is_heading_env[climbing_env_ids] = True

        # 4. 鍦ㄥ钩鍦帮紙flat锛夊湴褰笂鐢熸垚 35% 绾?Y 鍜?35% 绾?Z 鎸囦护锛屾帓闄ら潤姝㈢幆澧?
        if terrain_types is not None and self._flat_index != -1:
            is_flat = terrain_types == self._flat_index
            flat_env_ids = env_ids[is_flat[env_ids]]
            if len(flat_env_ids) > 0:
                active_flat_mask = ~self.is_standing_env[flat_env_ids]
                active_flat_ids = flat_env_ids[active_flat_mask]
                if len(active_flat_ids) > 0:
                    r = torch.empty(len(active_flat_ids), device=self.device).uniform_(0.0, 1.0)
                    y_only_mask = r < 0.35
                    z_only_mask = (r >= 0.35) & (r < 0.70)

                    y_only_ids = active_flat_ids[y_only_mask]
                    if len(y_only_ids) > 0:
                        self.vel_command_b[y_only_ids, 0] = 0.0  # x = 0
                        self.vel_command_b[y_only_ids, 2] = 0.0  # yaw = 0
                        if self.cfg.heading_command:
                            self.heading_target[y_only_ids] = self.wheelleg.data.heading_w[y_only_ids]
                            self.is_heading_env[y_only_ids] = True

                    z_only_ids = active_flat_ids[z_only_mask]
                    if len(z_only_ids) > 0:
                        self.vel_command_b[z_only_ids, 0] = 0.0  # x = 0
                        self.vel_command_b[z_only_ids, 1] = 0.0  # y = 0
                        if self.cfg.heading_command:
                            yaw_delta = torch.empty(len(z_only_ids), device=self.device).uniform_(
                                -self.cfg.yaw_only_heading_range, self.cfg.yaw_only_heading_range
                            )
                            self.heading_target[z_only_ids] = self.wheelleg.data.heading_w[z_only_ids] + yaw_delta
                            self.is_heading_env[z_only_ids] = True

    def _update_command(self) -> None:
        # 璋冪敤鍩虹被鐨勬瘡姝ユ洿鏂?
        super()._update_command()
        
        # 4. 姣忔鏇存柊鏃跺己鍔涚害鏉燂細鑻ュ浜庡彴闃?鐭鐖珮鍦板舰锛屽己鍒?y 杞存í绉绘寔缁负 0.0锛屽厑璁告俯鍜岀殑鍋忚埅绾犲亸瀵归綈鍙伴樁
        terrain = getattr(self._env.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if terrain_types is not None and len(self._climbing_indices) > 0:
            is_climbing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for idx in self._climbing_indices:
                is_climbing |= (terrain_types == idx)

            # 銆愬嚭鍙伴樁瀹炴椂閲嶉噰鏍枫€戯細妫€鏌ヤ粠鐖潯鐘舵€佸垰鍒氱寮€鐜鐨勬満鍣ㄤ汉锛岃繘琛屽嵆鏃舵寚浠ら噸閲囨牱
            left_climbing_mask = self.was_climbing & ~is_climbing
            if left_climbing_mask.any():
                left_climbing_ids = torch.where(left_climbing_mask)[0]
                self._resample_command(left_climbing_ids)

            # 銆愬叆鍙伴樁瀹炴椂鎴柇銆戯細瀵逛簬姝ｅ湪鐖ゼ姊?缈昏秺鐭鐨勭幆澧冿紝寮鸿鎴柇鍏舵í鍚戞í绉绘寚浠わ紝骞惰繘琛屾俯鍜屽亸鑸榻?
            climbing_env_ids = is_climbing.nonzero(as_tuple=False).flatten()
            if len(climbing_env_ids) > 0:
                self.vel_command_b[climbing_env_ids, 1] = 0.0
                # 馃専 鍏佽寰急鐨勫亸鑸籂鍋忥紝灏嗗亸鑸寚浠ら檺鍒跺湪娓╁拰鐨?[-0.3, 0.3] 鍖洪棿锛岄槻姝㈣繃搴︾敥灏撅紝浣嗕繚璇佽兘淇鑸悜
                self.vel_command_b[climbing_env_ids, 2] = torch.clip(
                    self.cfg.heading_control_stiffness * self.heading_error[climbing_env_ids],
                    min=-0.3,
                    max=0.3
                )
            self.was_climbing = is_climbing

        # 5. 楂橀€熶晶鍚戣В鑰︼紙閫傜敤浜庡钩鍦?鏂滃潯绛夋贩鍚堣矾闈級锛氬綋鍓嶈繘閫熷害 >= 0.8 m/s 鏃讹紝娓呯┖渚у悜鎸囦护锛岄槻姝㈤珮閫熺敥灏剧敥椋?
        high_speed_mask = self.vel_command_b[:, 0].abs() >= 0.8
        high_speed_ids = high_speed_mask.nonzero(as_tuple=False).flatten()
        if len(high_speed_ids) > 0:
            self.vel_command_b[high_speed_ids, 1] = 0.0


@dataclass(kw_only=True)
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformThresholdVelocityCommand
    yaw_only_heading_range: float = 1.0
    rel_lateral_envs: float = 0.0
    rel_yaw_envs: float = 0.0

"""HIK LMM 推理 服务端"""
import socket
import cv2
import time
import numpy as np
import struct
# from infer_elite_rela import *
# from infer_3B_chunk_20250910_2wristCam import *
from infer import *
import threading
import random
import traceback
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy

traj_ind = 0  # 全局变量，
last_gripper_state = 0.07


def transform(cur_pos, action_list):
    cur_mat = np.eye(4)
    rotation = R.from_euler("xyz", cur_pos[3:-1])
    cur_mat[:3, :3] = rotation.as_matrix()
    cur_mat[:3, 3] = cur_pos[:3]

    rotation = R.from_euler("xyz", action_list[3:-1])
    action_pose = np.eye(4)
    action_pose[:3, :3] = rotation.as_matrix()
    action_pose[:3, 3] = action_list[:3]
    next_pose = cur_mat @ action_pose

    euler = R.from_matrix(next_pose[:3, :3]).as_euler("xyz")
    next_pose_list = next_pose[:3, 3].tolist() + euler.tolist()
    next_pose_list.append(action_list[-1])
    return next_pose_list


# 实例化bin模型
# model = SYNC_MODEL_V1(MODEL_CFG)
model = MODEL(MODEL_CFG)


# 如果当前预测的state超出了工作空间，则有可能是预测异常点，保护执行并重启逃出异常点
def protect_state(src_state, state, output_text):
    temp_src_state = deepcopy(src_state)
    i = random.randint(0, 2)
    pro_bias = random.choice([-0.005, 0.005])
    if output_text == "pred_beyond":
        temp_src_state[i] += pro_bias
        print("occur protect_state")
        return temp_src_state
    else:
        return state


def LMM_Process(images, text, robot_state, refresh, register_img=[]):
    """
    images: list(numpy, ...)
    text: example: "pick the blue medium coffe cup on the bar and place it on the tray"
    robot_state: list()
    """
    print("In LMM Process")

    # if len(register_img) == 0:
    #     pass  # text = "pick up the camera module"
    # #    print("register_img!=None")
    # #    if "<IMG>" not in text:
    # #        print("please using <IMG> with register format")   #pick up <IMG>
    # #        return "error", robot_state, int(0), int(1),int(0),"None"
    # #
    # #    else:
    # #        print("register_img:",register_img.shape)
    # #        images.append(register_img)
    # else:
    #     print("register_img:", register_img.shape)
    #     images.append(register_img)

    if len(register_img) > 0:
        print("Recieved Register_img")

    try:
        global traj_ind
        # 如果要更新模型:
        if refresh:
            model.clear()
            traj_ind = 0
            # return "error", robot_state, int(0), int(1)  # TODO: elite修改"success", next_state, is_collision, is_term, is_rej, "None"
            return "error", robot_state, int(0), int(0), int(0), "None" 

        # 1. 读取图片&状态 TODO 相机的RGB\腕部分辨率\夹爪状态怎么传入？、图片按左肩、右肩、腕部传入，腕部视角默认倒转
        images = [i[:, :, ::-1] for i in images]  # BGR2RGB
        images = [PIL.Image.fromarray(i) for i in images]

        # flip
        # images[-2] = images[-2].rotate(180)  # 腕部视角旋转180°
        images[-1] = images[-1].rotate(180)  # 腕部视角旋转180°

        # state
        robot_state = list(robot_state)


        # # 只使用2个腕部相机
        # images = [images[-1], images[-2]]

        # # 只使用1个腕部相机
        # images = [images[-1], ]

        robot_state[-1] = 1 - robot_state[-1]  # 翻转夹爪状态
        # print(20 * "=")
        # print(robot_state)
        # print(states_text)
        images[0].save("/data1/VLA/baseline/0.jpg")
        images[1].save("/data1/VLA/baseline/1.jpg")
        images[2].save("/data1/VLA/baseline/2.jpg")
        images[3].save("/data1/VLA/baseline/3.jpg")

        # for i in range(len(images)):
        #     images[i].save("/data1/transfer/fhc/EBAI_VLA/EBAI_VLA_HEAD/ebai_vla_inference/%d.jpg" % i)  # wrist cam 1(up)

        # # TODO: 调试使用
        # img_pth = [
        #     r'/dataset/lizheyang/EBAI_2_1/BoxPick_NoArm/SamePoseObject1/287/armV2_rgb_wrist_1_8.jpg',
        #     r'/dataset/lizheyang/EBAI_2_1/BoxPick_NoArm/SamePoseObject1/287/armV2_rgb_wrist_2_8.jpg',
        # ]
        # imgs = [cv2.imread(i) for i in img_pth]  # 传入bgr
        # images = [i[:, :, ::-1] for i in imgs]  # BGR2RGB
        # images = [PIL.Image.fromarray(i) for i in images]
        # # ----------------------------------------------
        # -------------------------------------------------------------
        # temp_robot_state = deepcopy(robot_state)
        # temp_robot_state[-1] = 1 - temp_robot_state[-1]  # 已确定肯定要1-
        #
        # # 编写问题
        # states_text = model.mk_states(temp_robot_state)
        # # rgb_prompts = "rgb rear left view <IMG>, rgb wrist view <IMG>, rgb rear right view <IMG>"
        # rgb_prompts = "rgb rear left view <IMG>, rgb rear right view <IMG>, rgb wrist view <IMG>"
        # question = f"You are a {ROBOT_TYPE} robot using {CONTROL_TYPE}. robot's {rgb_prompts}, robot's previous {HISTORY_SIZE} state is: {states_text}. "
        # question += f"what action should robot take to {text} on the next keyframe? Answer the question in the format of {{translation x,y,z;rotation θ,ψ,φ;gripper angle; is collision detection required; has the task been completed;has the task been rejected}}, for example, {{128,138,154;127,122,142;007;yes;no;no}}"
        # dialogue_before = [["User", question], ['Assistant', '']]
        # print("in llm_server,the question is:", question)
        #
        # print("in llm_server the robot_state is:", robot_state)
        # print("in llm_server the state_text is:", states_text)
        # images[0].save("./left_0222.png")
        # images[1].save("./right_0222.png")
        # images[2].save("./wirst_0222.png")
        # if len(images) == 4:
        #     images[3].save("./register.png")
        #
        # # 提问至模型
        # ret = model.forward(images=images, dialogue_before=dialogue_before, max_new_tokens=256, act_dec=False)
        #
        # # 解析模型的输出 TODO
        # print("in llm server the model output is:", ret)
        # output_text, kact, is_collision, is_term, is_rej = model.decode_keyframe(ret)
        # # 修改夹爪方向
        # kact[-1] = 1 - kact[-1]
        #
        # if kact[-1] > 0.45:
        #     kact[-1] = 0.637
        #
        # print("in lmm server kact", kact)
        # print("output_text:", output_text)
        # print("kact", kact)
        # pred_next_state = transform(robot_state, kact)
        # print("pred_next_state", pred_next_state)
        # next_state = protect_state(robot_state, pred_next_state, output_text)
        # return output_text, next_state, is_collision, is_term, is_rej, "None"
        # ----------------------------------------------------
        time_start = time.time()
        print("LMM_Process: before model.forward")
        # print("input task:", text)
        # print('input state', robot_state)
        # 提问至模型
        # robot_state = None
        # ret = model.infer(traj_ind, images, text, robot_state)
        raw_ret = model.forward(images, text, robot_state)
        print("LMM_Process: after model.forward")
        ret = np.sum(np.array(raw_ret)[:1, :6],axis=0).tolist()
        ret.append(raw_ret[0][-1])
        ret = transform(robot_state, ret)
        print(ret)
        traj_ind += 1
        ret[-1] = 1 - ret[-1]  # 翻转夹爪状态

        # # 临时策略
        # global last_gripper_state
        # shimit_up = 0.27
        # gripper_limit = (0.12, 0.21)
        # print("original gripper: %.4f" % ret[-1])
        # if (last_gripper_state < gripper_limit[1]) and (ret[-1] > gripper_limit[1]):
        #     ret[-1] = shimit_up
        # elif (last_gripper_state > gripper_limit[1]) and (ret[-1] > gripper_limit[0]):
        #     ret[-1] = shimit_up
        # last_gripper_state = ret[-1]
        # print("trans    gripper: %.4f" % ret[-1])

        time_end = time.time()
        print(f"time-cost: {time_end - time_start}")
        next_state = ret
        is_collision = 0  # 是否碰撞检测
        is_term = 0
        is_rej = 0
        return "success", next_state, is_collision, is_term, is_rej, "None"  # transform(robot_state, ret[1]), int(0), int(0)


    except Exception as e:
        print("process............... ", e)
        # Print full stack trace to locate where unexpected kwargs originate.
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return "error", robot_state, int(0), int(1), int(0), "None"


def receive_data(conn, length):
    """接收固定长度的数据"""
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data


def handle_client(conn):
    """处理客户端请求"""
    # 接收left图像
    while True:
        try:
            regedit_length = struct.unpack(">I", receive_data(conn, 4))[0]
            if regedit_length > 0:
                regedit_data = receive_data(conn, regedit_length)
                regedit_img = cv2.imdecode(np.frombuffer(regedit_data, np.uint8), cv2.IMREAD_COLOR)
            else:
                regedit_img = []
            
            # 接收top图像
            img1_length = struct.unpack(">I", receive_data(conn, 4))[0]
            img1_data = receive_data(conn, img1_length)
            img1 = cv2.imdecode(np.frombuffer(img1_data, np.uint8), cv2.IMREAD_COLOR)

            # 接收chest图像
            img2_length = struct.unpack(">I", receive_data(conn, 4))[0]
            img2_data = receive_data(conn, img2_length)
            img2 = cv2.imdecode(np.frombuffer(img2_data, np.uint8), cv2.IMREAD_COLOR)

            # 接收wrist1图像
            img3_length = struct.unpack(">I", receive_data(conn, 4))[0]
            img3_data = receive_data(conn, img3_length)
            img3 = cv2.imdecode(np.frombuffer(img3_data, np.uint8), cv2.IMREAD_COLOR)

            # 接收wrist2图像
            img4_length = struct.unpack(">I", receive_data(conn, 4))[0]
            img4_data = receive_data(conn, img4_length)
            img4 = cv2.imdecode(np.frombuffer(img4_data, np.uint8), cv2.IMREAD_COLOR)

            # 接收语言指令
            text_length = struct.unpack(">I", receive_data(conn, 4))[0]
            text = receive_data(conn, text_length).decode('utf-8')

            # 接收机械臂当前状态 x, y, z, rx，ry, rz, gripper
            robot_data = receive_data(conn, 7 * 4)
            robot_list = list(struct.unpack(">7f", robot_data))

            # # 接收refresh
            # refresh = struct.unpack(">I", receive_data(conn, 4))[0]

            # 调用处理逻辑
            processed_text, robot_action, col_flag, term_flag, reject_flag, reason = LMM_Process(
                [img1, img2, img3, img4], text, robot_list, 0, regedit_img
            )
            # client端如下
            # flag, processed_text, action_list, term_flag, reject_flag = self.get_response()
            if processed_text != 'success':
                robot_action = [0.0] * 7
                
            print('test: ', text)
            print("robot_list:  ", robot_list)
            print("robot_action ", robot_action)
            print("col_flag ", col_flag)
            print("term_flag ", term_flag)
            print("reject_flag ", reject_flag)

            # 发送action
            conn.sendall(struct.pack(">7f", *robot_action))

            # 发送终止flag
            conn.sendall(struct.pack(">I", term_flag))

            # send reject
            conn.sendall(struct.pack(">I", reject_flag))

            # 发送语言反馈
            processed_text_bytes = processed_text.encode('utf-8')
            conn.sendall(struct.pack(">I", len(processed_text_bytes)))
            conn.sendall(processed_text_bytes)
        except Exception as e:
            print("handle_client..............", e)
            continue

    conn.close()


def start_server(host='0.0.0.0', port=5000):
    """启动服务器"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)

    while True:
        try:
            print(f"Server listening on {host}:{port}")
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            client_thread = threading.Thread(target=handle_client, args=(conn,), daemon=True)
            client_thread.start()
        except Exception as e:
            print(e)
            pass


def infer_debug():
    # TODO: 临时测试使用
    img_pth = ["/dataset/lizheyang/EBAI_2_1/EvaPnp/rubber500_1217/AlignData/101/rgb_top_1.jpg",
               "/dataset/lizheyang/EBAI_2_1/EvaPnp/rubber500_1217/AlignData/101/rgb_chest_1.jpg",
               "/dataset/lizheyang/EBAI_2_1/EvaPnp/rubber500_1217/AlignData/101/rgb_wrist_1_1.jpg",
               "/dataset/lizheyang/EBAI_2_1/EvaPnp/rubber500_1217/AlignData/101/rgb_wrist_2_1.jpg"]
    # img_pth = [
    #     r'/dataset/lizheyang/EBAI_2_1/BoxPick_NoArm/SamePoseObject1/287/armV2_rgb_wrist_1_8.jpg',
    #     r'/dataset/lizheyang/EBAI_2_1/BoxPick_NoArm/SamePoseObject1/287/armV2_rgb_wrist_2_8.jpg',
    # ]
    imgs = [cv2.imread(i) for i in img_pth]  # 传入bgr
    # task = "pick up the eva material from the carton"
    task = "pick the workpiece in the cardboard box and place it in the grid fixture tray"
    # robot_state = [0.29172202, -0.00546935,  0.07605619, -3.1212234,  0.02093441, 0.12358301,  0.993]
    robot_state = [
        -0.1576425121128559,
        -0.17011539173126222,
        0.29576025223731994,
        -3.1094397391567523,
        -0.04102535193600976,
        1.770517303904441,
        1.0 - 0
      ]
    for i in range(10):
        robot_state[0] = robot_state[0] + 0.01
        ret = LMM_Process(images=imgs, text=task, robot_state=robot_state, refresh=0)
        print(ret)


if __name__ == "__main__":
    start_server()
    # infer_debug()

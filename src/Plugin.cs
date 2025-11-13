using BepInEx;
using BepInEx.Logging;
using System.Security.Permissions;
using UnityEngine; // Time.deltaTime 사용을 위해 필요
using System.Collections.Generic; // List 사용을 위해 필요
using System.IO;
using System;
using System.Reflection;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using Newtonsoft.Json;           // JSON 직렬화를 위해 (필요한 외부 라이브러리)
using System.Linq;

// Unity/Game 타입에 대한 가상의 using 구문 (타겟 게임에 맞게 변경 필요)
// using RWCustom; 
// using RainWorldGame; 
// using Menu; 

// Allows access to private members
#pragma warning disable CS0618
[assembly: SecurityPermission(SecurityAction.RequestMinimum, SkipVerification = true)]
#pragma warning restore CS0618

namespace RLProject;

[BepInPlugin("minseon23.rl", "RLProject", "0.1.0")]
sealed class Plugin : BaseUnityPlugin
{
    public static new ManualLogSource Logger;
    bool IsInit;
    bool IsTrain = true;
    bool endflag = false;
    float timescale = 1.0f;
    float updatetime = 0.2f;
    bool enable_connect = true;
    bool body_set = false;
    Vector2 newPos = new Vector2(680, 200);
    int action = 4;
    bool shouldRestartNextFrame;

    // --- 타이머 및 상수 필드 추가 ---
    private float logTimer = 0f;
    private float jumpTimer = 0f;
    private const float logInterval = 0.1f; // 1초 간격

    private TcpListener tcpListener;
    private Thread listenThread;
    private TcpClient client;
    private NetworkStream stream;

    private const int PORT = 50000; // 사용할 포트 번호
    public void OnEnable()
    {
        Logger = base.Logger;
        On.RainWorld.OnModsInit += OnModsInit;

        // --- Player.Update 후킹 추가 ---
        // Player 인스턴스(self)를 매 프레임 얻을 수 있습니다.
        On.Player.UpdateMSC += Player_UpdateMSC;
    }

    private void OnModsInit(On.RainWorld.orig_OnModsInit orig, RainWorld self)
    {
        orig(self);

        if (IsInit) return;
        IsInit = true;
        
        Logger.LogDebug("RLProject Initialized.");

        if (!enable_connect) return;

        listenThread = new Thread(new ThreadStart(StartTcpServer));
        listenThread.IsBackground = true; // 게임 종료 시 스레드 자동 종료
        listenThread.Start();
    }

    private void StartTcpServer()
    {
        try
        {
            tcpListener = new TcpListener(System.Net.IPAddress.Parse("127.0.0.1"), PORT);
            tcpListener.Start();
            Logger.LogInfo($"[Server] Waiting for Python client on port {PORT}...");

            // 클라이언트 연결 수락 (블로킹)
            client = tcpListener.AcceptTcpClient();
            stream = client.GetStream();
            Logger.LogInfo("[Server] Python client connected successfully!");
        }
        catch (Exception ex)
        {
            Logger.LogError($"[Server Error] {ex.Message}");
        }
        try
        {
            byte[] confirmationBytes = new byte[4];
            int bytesRead = stream.Read(confirmationBytes, 0, confirmationBytes.Length);
            
            if (bytesRead > 0)
            {
                int confirmationValue = BitConverter.ToInt32(confirmationBytes, 0);
                Logger.LogInfo($"[C#] Received confirmation signal: {confirmationValue}");
                timescale = confirmationValue;
                Time.timeScale = timescale;
            }
            else
            {
                Logger.LogWarning("[C#] Received 0 bytes (connection closed).");

            }
        }
        catch
        {
            Logger.LogDebug("Timescale Initialization has failed");
        }
    }

    // --- 핵심 로직: Player.Update 메서드 후킹 ---
    // (Player, Room, PhysicalObject 타입은 게임의 클래스명을 따릅니다.)
    private void Player_UpdateMSC(On.Player.orig_UpdateMSC orig, Player self)
    {
        if (shouldRestartNextFrame)
        {
            // RestartGame()을 호출하고 플래그 초기화
            self.abstractCreature.Room.realizedRoom.game.RestartGame();
            shouldRestartNextFrame = false;
            // endflag 및 body_set 초기화는 재시작 시점에 Player.ctor에서 처리하는 것이 더 안전합니다.
            // 여기서는 플래그만 초기화
            endflag = false;
            body_set = false;
            orig(self);
            return; // 현재 프레임은 여기서 종료하고 재시작으로 넘어갑니다.
        }
        // 1. 타이머 업데이트 및 체크
        logTimer += Time.deltaTime; // Unity의 마지막 프레임 이후 경과 시간
        jumpTimer += Time.deltaTime;
        if (logTimer >= updatetime)
        {
            if (!body_set)
            {
                position_setting(self);
                body_set = true;
            }
            logTimer = 0f; // 타이머 재설정

            // 점프를 원할 때
            // self.wantToJump = 100;

            // --- A. 플레이어 위치 (x, y) 추출 및 출력 ---
            // self.mainBodyChunk.pos.x/y는 BepInEx 환경에서 게임의 타입을 사용해야 합니다.
            float playerX = self.mainBodyChunk.pos.x;
            float playerY = self.mainBodyChunk.pos.y;
            Logger.LogInfo($"[RL State] Player Pos: ({playerX:F2}, {playerY:F2})");

            // --- B. 존재하는 모든 생물의 위치 (x, y) 추출 및 출력 ---

            // 1) Room 인스턴스 획득 (발견된 경로 사용)
            // (AbstractCreature와 Room은 게임 내에서 정의된 클래스여야 합니다.)
            AbstractRoom currentRoom = self.abstractCreature.Room;

            if (self.dead == true || endflag == true || self.slatedForDeletetion) // <--- slatedForDeletetion 추가
            {
                // 게임 종료/클리어 감지 시, 즉시 재시작 대신 예약을 합니다.
                shouldRestartNextFrame = true; // 다음 프레임에 재시작 예약
                action = SendObservationAndReceiveAction(self);
            }
            else
            {
                // 정상적인 학습 루프
                action = SendObservationAndReceiveAction(self);

                Logger.LogDebug(action);
                
                if (action < 0)
                {
                    endflag = true;
                }
                if (action == 5)
                {
                    self.wantToJump = 100;
                    jumpTimer = 0f;
                }
            }
        }
            //=== action space ===
            //0 : left
            //1 : right
            //2 : down
            //3 : up
            //4 : stay
            //5 : jump

        switch (action)
        {
            case 0: self.input[0].x = -1; break; // left
            case 1: self.input[0].x = 1; break;  // right
            case 2: self.input[0].y = -1; break; // down
            case 3: self.input[0].y = 1; break;  // up
            case 4: break;
            break;
        // -1 (종료 신호) 등은 default에서 처리
        default: break;
        }
        if (jumpTimer < 1.0f)
        {
            self.input[0].jmp = true;
        }
        orig(self); // 원본 Player.Update 메서드 호출 (필수)
    }

    private void position_setting(Player self)
    {

        // 3. 모든 BodyChunk 순회 및 위치/속도 설정
        if (self.bodyChunks != null)
        {
            foreach (BodyChunk chunk in self.bodyChunks)
            {
                // 위치 설정 (teleport)
                chunk.pos = newPos;
                // 직전 위치도 동일하게 설정 (안정적인 텔레포트 효과)
                chunk.lastPos = newPos;
                    
                // 속도 초기화 (정지 상태로 시작)
                chunk.vel = Vector2.zero; // Vector2.zero는 (0f, 0f)와 동일합니다.
            }

            Plugin.Logger.LogInfo($"[RL] Player and all BodyChunks initialized at ({newPos.x}, {newPos.y}) with zero velocity.");
        }
        else
        {
            Plugin.Logger.LogWarning("[RL] Player bodyChunks array was null during ctor.");
        }
    }
    
    private List<float> GetCurrentObservationData(Player self)
    {
        // --- 1. 플레이어 위치 추출 ---
        float playerX = self.mainBodyChunk.pos.x;
        float playerY = self.mainBodyChunk.pos.y;
        List<float> retlist = new List<float>();
        retlist.Add(playerX);
        retlist.Add(playerY);

        // --- 2. 모든 생물 위치 추출 및 저장 ---
        var creaturePositions = new List<float[]>();

        AbstractRoom currentRoom = self.abstractCreature.Room;

        if (currentRoom != null)
        {
            // 2) 물리 개체 리스트에 접근 (physicalObjects[0]에 생물이 있다고 가정)
            // (PhysicalObject는 게임 내에서 정의된 클래스여야 합니다.)
            List<AbstractCreature> creatureList = currentRoom.creatures;

            Logger.LogInfo($"[RL State] --- Total Creatures: {creatureList.Count} ---");

            foreach (AbstractCreature aCreature in creatureList)
            {
                // 실체화된 생물체인지 확인
                if (aCreature.realizedCreature != null)
                {
                    // Player 자신은 추적 목록에서 제외
                    if (aCreature.realizedCreature == self)
                    {
                        continue;
                    }

                    Creature realizedCreature = aCreature.realizedCreature;

                    // 위치 정보 추출
                    float creatureX = realizedCreature.mainBodyChunk.pos.x;
                    float creatureY = realizedCreature.mainBodyChunk.pos.y;

                    creaturePositions.Add(new float[] { creatureX, creatureY });
                    retlist.Add(creatureX);
                    retlist.Add(creatureY);
                }
            }
        }

        return retlist;
    }

    private int SendObservationAndReceiveAction(Player self)
    {
        if (!enable_connect) return 400;
        if (stream == null || !client.Connected) return 404;

        try
        {
            float[] dataToSend;
            if (shouldRestartNextFrame)
            {
                dataToSend = new float[] {
                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
                };
            }
            else
            {
                List<float> obsData = GetCurrentObservationData(self);
                dataToSend = new float[] {
                    obsData[0], obsData[1], obsData[2], obsData[3], obsData[4], obsData[5], obsData[6], obsData[7]
                };
            }


            // 데이터를 바이트 배열로 변환합니다. (Float 하나당 4바이트)
            byte[] payloadBytes = new byte[dataToSend.Length * 4];
            Buffer.BlockCopy(dataToSend, 0, payloadBytes, 0, payloadBytes.Length);

            // 2. 데이터 전송 (길이 헤더 [4바이트] + 실제 데이터)
            byte[] lengthHeader = BitConverter.GetBytes(payloadBytes.Length);

            stream.Write(lengthHeader, 0, lengthHeader.Length); // 길이 헤더 전송
            stream.Write(payloadBytes, 0, payloadBytes.Length);   // 실제 데이터 전송
            stream.Flush();

            Logger.LogInfo($"[C#] Sent {dataToSend.Length} floats ({payloadBytes.Length} bytes).");

            // 3. Python으로부터 1 (확인 신호) 수신
            byte[] confirmationBytes = new byte[4]; // int는 4바이트
            int bytesRead = stream.Read(confirmationBytes, 0, confirmationBytes.Length);

            if (bytesRead > 0)
            {
                int confirmationValue = BitConverter.ToInt32(confirmationBytes, 0);
                Logger.LogInfo($"[C#] Received confirmation signal: {confirmationValue}");

                return confirmationValue;
            }
            else
            {
                Logger.LogWarning("[C#] Received 0 bytes (connection closed).");
                return -1;
            }
        }
        catch (IOException ex) when (ex.InnerException is SocketException)
        {
            Logger.LogError("[C#] Socket Error during communication. Client likely disconnected.");
            // 여기서 클라이언트 연결 종료 처리를 할 수 있습니다.
            stream.Close();
            client.Close();
            stream = null;
            return -1;
        }
        catch (Exception ex)
        {
            Logger.LogError($"[C#] Communication Error: {ex.Message}");
            return -1;
        }
    }
}
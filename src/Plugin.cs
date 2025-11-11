using BepInEx;
using BepInEx.Logging;
using System.Security.Permissions;
using UnityEngine; // Time.deltaTime 사용을 위해 필요
using System.Collections.Generic; // List 사용을 위해 필요

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

    // --- 타이머 및 상수 필드 추가 ---
    private float logTimer = 0f;
    private const float logInterval = 1.0f; // 1초 간격

    public void OnEnable()
    {
        Logger = base.Logger;
        On.RainWorld.OnModsInit += OnModsInit;

        // --- Player.Update 후킹 추가 ---
        // Player 인스턴스(self)를 매 프레임 얻을 수 있습니다.
        On.Player.Update += Player_Update;
    }

    private void OnModsInit(On.RainWorld.orig_OnModsInit orig, RainWorld self)
    {
        orig(self);

        if (IsInit) return;
        IsInit = true;

        Logger.LogDebug("RLProject Initialized.");
    }

    // --- 핵심 로직: Player.Update 메서드 후킹 ---
    // (Player, Room, PhysicalObject 타입은 게임의 클래스명을 따릅니다.)
    private void Player_Update(On.Player.orig_Update orig, Player self, bool eu)
    {
        orig(self, eu); // 원본 Player.Update 메서드 호출 (필수)

        // 1. 타이머 업데이트 및 체크
        logTimer += Time.deltaTime; // Unity의 마지막 프레임 이후 경과 시간

        if (logTimer >= logInterval)
        {
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

            if (currentRoom != null)
            {
                // 2) 물리 개체 리스트에 접근 (physicalObjects[0]에 생물이 있다고 가정)
                // (PhysicalObject는 게임 내에서 정의된 클래스여야 합니다.)
                List<AbstractCreature> creatureList = currentRoom.creatures;

                Logger.LogInfo($"[RL State] --- Total Creatures: {creatureList.Count} ---");

                int creatureIndex = 0;
                foreach (AbstractCreature aCreature in creatureList)
                {
                    // 2) 추상 생물체가 현재 방에 '실체화(Realized)'되었는지 확인
                    if (aCreature.realizedCreature != null)
                    {
                        // Player 자신은 추적 목록에서 제외 (선택 사항)
                        if (aCreature.realizedCreature == self)
                        {
                            // Logger.LogInfo($"[RL State] Player: ({self.mainBodyChunk.pos.x:F2}, {self.mainBodyChunk.pos.y:F2}) (Skipping Player in Creature List)");
                            continue;
                        }

                        // 3) 실체화된 Creature 객체 (Lizard, 등)에서 위치 정보 추출
                        Creature realizedCreature = aCreature.realizedCreature;

                        // mainBodyChunk는 Creature 클래스에 존재합니다.
                        float creatureX = realizedCreature.mainBodyChunk.pos.x;
                        float creatureY = realizedCreature.mainBodyChunk.pos.y;

                        Logger.LogInfo($"[RL State] Creature {creatureIndex}: ({creatureX:F2}, {creatureY:F2})");
                        creatureIndex++;
                    }
                }
            }
            else
            {
                Logger.LogWarning("[RL State] Player is not currently loaded in a Room.");
            }
        }
    }
}
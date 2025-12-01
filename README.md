# 사용 방법

1. requirements.txt가 있는 경로에서 코드 실행 : 
`pip install -r requirements.txt`
2. 게임 홈 경로에서 BepInex/plugins 폴더로 이동
(혹은 `C:\Program Files (x86)\Steam\steamapps\common\Rain World\BepInEx\plugins` 경로로 이동)
해당 폴더에 `RLProject.dll` 복사
3. [main.py](http://main.py/) 실행 및 게임 실행
-게임과 파이썬 파일을 별도로 실행하면 자동으로 연결된다.
(게임이 완전히 실행되어도 연결이 거절되면 dll 파일이 잘 복사되어 있는지 다시 확인한다.)
4. 알고리즘 및 running speed 설정 : 
running speed만큼 게임 속도를 배속한다. 학습을 가속하나 오류 발생 가능성도 올라간다.

### 게임 안에서
아레나 -> 샌드박스에서 아래와 같이 세팅
<img width="1368" height="807" alt="Image" src="https://github.com/user-attachments/assets/84e1a198-78f5-41e9-b9f6-77ab8188f056" />
<img width="1368" height="807" alt="Image" src="https://github.com/user-attachments/assets/9c80dd3f-441e-4b03-9607-e2a427ca40c6" />

Use this template on GitHub or just [download the code](https://github.com/alduris/TemplateMod/archive/refs/heads/master.zip), whichever is easiest.

Rename `src/TestMod.csproj`, then edit `mod/modinfo.json` and `src/Plugin.cs` to customize your mod.

See [the modding wiki](https://rainworldmodding.miraheze.org/wiki/Downpour_Reference/Mod_Directories) for `modinfo.json` documentation.

To update your mod to work in future updates, replace `PUBLIC-Assembly-CSharp.dll` and `HOOKS-Assembly-CSharp.dll` with the equivalents found in `Rain World/BepInEx/utils` and `Rain World/BepInEx/plugins` as well as `Assembly-CSharp-firstpass.dll` found in `Rain World/RainWorld_Data/Managed`.

Download requirements.txt with :

`pip install -r requirements.txt`

핵심 파일 : src 폴더

rainworld_connector : 게임과 서버 연결

RLProject.sln : 솔루션 파일

Plugins.cs : 게임 상태 전송 & 행동 코드, 서버 연결 코드


Use this template on GitHub or just [download the code](https://github.com/alduris/TemplateMod/archive/refs/heads/master.zip), whichever is easiest.

Rename `src/TestMod.csproj`, then edit `mod/modinfo.json` and `src/Plugin.cs` to customize your mod.

See [the modding wiki](https://rainworldmodding.miraheze.org/wiki/Downpour_Reference/Mod_Directories) for `modinfo.json` documentation.

To update your mod to work in future updates, replace `PUBLIC-Assembly-CSharp.dll` and `HOOKS-Assembly-CSharp.dll` with the equivalents found in `Rain World/BepInEx/utils` and `Rain World/BepInEx/plugins` as well as `Assembly-CSharp-firstpass.dll` found in `Rain World/RainWorld_Data/Managed`.

Download requirements.txt with :

`pip install requirements.txt`

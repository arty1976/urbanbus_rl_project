# Skill 06 — Windows PowerShell + Git + artifact hygiene

## 목적

Windows PowerShell(PowerShell=마이크로소프트 명령행 셸) 환경에서 연구 코드, 테스트, 문서, artifact 산출물을 안전하게 관리한다.

이 스킬은 `.gitignore`에 걸린 `artifacts/`를 실수로 커밋하지 않고, Step별 코드/문서만 정리해서 GitHub에 push하는 작업에 사용한다.

## 언제 사용하나

- 여러 Step을 만든 뒤 커밋이 섞였을 때
- `git add .`를 하면 위험한 상태일 때
- `artifacts/`가 `.gitignore`에 막혔을 때
- 로컬 커밋과 GitHub push 상태를 확인해야 할 때
- Windows에서 `Remove-Item` 또는 `--clean-output`이 Access denied를 낼 때

## Git 상태 점검

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"

git remote -v
git branch --show-current
git fetch origin
git status -sb
git log --oneline --decorate --left-right --graph origin/main...HEAD
```

해석:

```text
> 로 시작하는 커밋 = 로컬에는 있지만 GitHub에는 아직 push 안 됨
< 로 시작하는 커밋 = GitHub에는 있지만 로컬에는 없음
```

## Step별 커밋 원칙

`git add .` 금지.

항상 Step별로 명시적으로 add한다.

예:

```powershell
git add `
  .\05_training\adapters\some_step.py `
  .\05_training\adapters\test_some_step.py `
  .\05_training\adapters\some_step.md

git commit -m "Add some step scaffold"
```

## artifacts 처리 원칙

기본 원칙:

```text
artifacts/는 커밋하지 않는다.
```

이유:

```text
대부분 재생성 가능
용량 증가
결과 파일과 코드 변경이 섞이면 diff가 지저분해짐
```

`.gitignore` 경고가 나와도 정상이다.

```text
The following paths are ignored by one of your .gitignore files:
artifacts
```

정말 강제로 넣어야 할 때만:

```powershell
git add -f .\artifacts\some_path\
```

하지만 일반적으로 권장하지 않는다.

## push 절차

```powershell
git status -sb
git log --oneline --decorate origin/main..HEAD
git push origin main
git fetch origin
git log --oneline --decorate --left-right --graph origin/main...HEAD
```

정상 상태:

```text
origin/main...HEAD 비교 로그가 비어 있음
```

## Windows Access denied 처리

증상:

```text
[WinError 5] 액세스가 거부되었습니다
```

대응:

```powershell
$Target = ".\artifacts\daegu_bis_api_audit\some_output"

attrib -R "$Target\*" /S /D
Remove-Item -LiteralPath $Target -Recurse -Force
```

그래도 실패하면:

```text
VS Code
Excel
탐색기 미리보기
Python process
Jupyter
```

중 해당 파일을 잡고 있는 프로그램을 닫는다.

가능하면 `--clean-output` 없이 overwrite 방식으로 재실행한다.

## py_compile + self-test 패턴

권장 실행 순서:

```powershell
python -m py_compile `
  path\to\script.py `
  path\to\test_script.py

python path\to\test_script.py
python path\to\script.py
```

## 완료 기준

- Step별 커밋 완료
- artifacts는 커밋하지 않음
- origin/main과 HEAD 차이가 없음
- push 완료 확인
- Windows 파일 잠금 문제 대응 기록

## 에이전트용 프롬프트

```text
Windows PowerShell 환경에서 urbanbus_rl_project의 Step별 변경을 안전하게 커밋하고 push하는 절차를 안내하라.
git add .는 금지하고, Step별 py/test/md 파일만 명시적으로 add하라.
artifacts/는 .gitignore 대상이므로 기본적으로 커밋하지 말라.
origin/main...HEAD로 push 여부를 확인하고, Windows Access denied가 발생하면 파일 잠금과 Remove-Item 대안을 안내하라.
```

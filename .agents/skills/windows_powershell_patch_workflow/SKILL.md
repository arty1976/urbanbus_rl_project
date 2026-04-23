---
name: windows_powershell_patch_workflow
description: "사용자가 파일을 직접 편집하지 않고 PowerShell 명령어 붙여넣기(Patch)로 파일을 수정하는 작업 패턴"
---

# windows_powershell_patch_workflow

## 언제 이 skill을 써야 하는가
- Windows 환경에서 에이전트가 코드를 제안할 때
- 사용자의 수동 편집을 최소화하고, 콘솔에 붙여넣기만으로 파일 내용 수정(줄 단위 수정, 파일 전체 덮어쓰기)을 해야 할 때

## 원칙
**앞으로 모든 코드 수정안은 사용자가 직접 에디터를 열고 고치게 하지 말고, 이 skill의 "붙여넣기형 patch" 명령어로 제공해야 합니다.**

## 실행 순서 및 방식

### 1. 파일 전체 재작성 (Here-String 방식)
전체 파일을 새로 작성하거나 완전히 교체할 때 사용합니다. Literal Here-String(`@' ... '@`)을 사용하여 파이썬이나 bash 스크립트의 `$var` 등의 기호가 PowerShell 변수로 오작동하지 않게 보호합니다.

```powershell
$content = @'
def main():
    print("hello world")
    
if __name__ == "__main__":
    main()
'@
$content | Set-Content -Path "test.py" -Encoding UTF8
```

### 2. 줄 단위 부분 수정 (Get-Content & -replace)
특정 텍스트만 찾아서 다른 텍스트로 치환할 때 사용합니다. 정규표현식이 적용되므로 특수문자에 주의해야 합니다. `-Raw` 파라미터 사용 여부를 구분해야 합니다.

**줄 단위 수정 (일반적인 경우):**
```powershell
(Get-Content -Path "test.py") -replace 'print\("hello world"\)', 'print("hello urbanbus")' | Set-Content -Path "test.py" -Encoding UTF8
```

**여러 줄 또는 특수한 경우 (-Raw 활용):**
단일 문자열로 읽어서 줄바꿈 기호를 포함한 큰 블록을 정규표현식으로 교체할 때는 `-Raw`를 사용합니다.
```powershell
(Get-Content -Path "test.py" -Raw) -replace 'def main\(\):', 'def main(args):' | Set-Content -Path "test.py" -Encoding UTF8
```

## 실패 시 복구법 및 트러블슈팅 사례

### 1. Here-String 중첩으로 인한 변수 Null 문제
- **증상**: 파이썬/Bash 쉘 등 `$var`를 사용하는 코드를 겹따옴표 기반 Here-String(`@" ... "@`) 안에 넣으면 PowerShell이 이를 자신의 변수로 평가해버려서 빈 값으로 치환됨.
- **해결법**: 파일 생성/수정 시에는 반드시 홑따옴표 기반의 Literal Here-String(`@' ... '@`)을 사용합니다.

### 2. UTF-8 BOM 이슈
- **증상**: 스크립트 실행, DB 입력(SQL) 시 `\ufeff` 같은 BOM 문자가 맨 앞에 포함되어 `Syntax Error`를 유발함.
- **해결법**: `Set-Content` 시 `Out-File -Encoding utf8` 보다는 최신 PowerShell에서는 `Set-Content -Encoding UTF8` (또는 PowerShell Core에선 utf8NoBOM이 기본)을 사용하여 BOM 이슈를 최소화합니다. 특히 JSON이나 SQL을 저장할 때 조심해야 합니다. JSON의 경우 `utf-8-sig`로 읽도록 파이썬을 고치는 것도 좋은 우회법입니다.

### 3. $MyInvocation.MyCommand.Path 문제
- **증상**: 파이썬/PS 스크립트를 클립보드에 붙여넣어 실행 시, 현재 스크립트 파일의 경로를 동적으로 참조하는 `$MyInvocation.MyCommand.Path`가 값을 반환하지 않아 에러가 발생.
- **해결법**: 대화형 프롬프트(REPL)에 붙여넣을 때에는 파일 기반 내장 변수가 동작하지 않으므로, 테스트 용도라면 `$PWD` 등 현재 디렉터리 기준 경로를 명시적으로 쓰게 유도하거나, 먼저 파일로 저장한 후 실행하도록 안내합니다.

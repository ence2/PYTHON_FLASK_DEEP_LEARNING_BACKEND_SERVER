import sys
import re
import soynlp
import hgtk
import time
from collections import deque
from datetime import datetime, timedelta
import spacing as Spacing

hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
isChosung = re.compile('[ㄱ-ㅣ]+')
MAX_STRING_SIZE = 179

def IsChosungWord(char):
    return len(isChosung.sub('', char)) == 0

def IsAbusedLang(text):
    if text == "ㅅㅂ":
        return True
    
    if text == "ㅆㅂ":
        return True

    if text == "ㅅㅃ":
        return True

    if text == "ㅆㅃ":
        return True

    if text == "ㄳㄲ":
        return True

    if text == "ㅅㄲ":
        return True

    if text == "ㅂㅅ":
        return True

    if text == "ㅃㅅ":
        return True

    if text == "ㄲㅈ":
        return True

    if text == "ㄷㅊ":
        return True

    return False

def PreprocessString(text):
    chosungYokCheck = False
    temp = ''
    for c in text:
        if c == 'l' or c == 'L' or c == '|' or c == 'i' or c == 'I' or c == '1':
            temp += 'ㅣ'
        else:
            temp += c

    text = temp

    text = hangul.sub('', text)
    text = soynlp.normalizer.only_hangle(text)

    # text = soynlp.normalizer.emoticon_normalize(text, 1)
    text = soynlp.normalizer.repeat_normalize(text, 1)

    splited = text.split(' ')
    final_text = []
    i = -1
    for string in splited:
        i += 1
        if (string == '') or (string is None): continue

        addSpace = False
        # 현재 string의 마지막 글자와 뒤에 첫 글자만 비교        
        # 마지막이 초성이니?

        if IsChosungWord(string[len(string) - 1]):
            # 부모 리스트 마지막이 아니면 실행
            if i < len(splited) - 1:
                # 다음 데이터의 첫 글자가 초성이 아닐 경우 띄어 쓰기
                if False == IsChosungWord(splited[i + 1][0]):
                    addSpace = True
        else:
            addSpace = True

        if addSpace:
            string += " "

        final_text.append(string)

    # 1. 만들어진 대상 띄어쓰기 단위로 쪼갬
    # 2. for문 돌면서 대상 초성포함 string 단어로 만든 후 실패한 초성 전부 제거
    # 3. 띄어쓰기 단위로 합침 => ' '.join(final_text)
    target = ''.join(final_text).split(' ')
    final_text = []
    for string in target:
        # 한글자 이상이면 조합 시도
        if len(string) > 1:
            q=deque()
            deqSize = 0
            for d in string:
                q.append(d)
                deqSize += 1
            
            result = ''
            # 글자 Compose
            while deqSize > 0:
                char = q.popleft()
                deqSize -= 1
                if False == IsChosungWord(char):
                    result += char
                    continue
                
                if char == 'ㅄ':
                    chosungYokCheck = True
                    result += char

                # 하나 꺼냈는데 꺼낼 게 없으면 끝냄
                if deqSize == 0:
                    break

                char2 = q.popleft()
                deqSize -= 1
                if False == IsChosungWord(char2):
                    result += char2
                    continue

                tempStr = str(char + char2)
                if IsAbusedLang(tempStr):
                    chosungYokCheck = True
                    result += tempStr
                    continue

                composed = ""
                try:
                    composed = hgtk.letter.compose(char, char2)    
                except Exception:
                    composed = str(char + char2)
                
                if composed == tempStr:
                    if IsAbusedLang(tempStr):
                        result += tempStr
                        continue

                    q.appendleft(char2)
                    deqSize += 1
                    continue

                result += composed

            final_text.append(result)
            continue

        if IsChosungWord(string) == True:
            continue
        # string = hgtk.text.compose(string)
        final_text.append(string)

    final_result = []
    for string in final_text:
        if (string == '') or (string is None): continue
        final_result.append(string)
    
    end_result = ' '.join(final_result)
    # if len(end_result) <= 6:
    #     return end_result

    if len(end_result) >= MAX_STRING_SIZE:
        return ""
    
    # 실제 서비스 구 현 할 때 하단 조건문 제거해야함
    # 시스템에서 필터링 가능한 것은 라벨링에서 제외 하려고 만든 조건문임
    # if chosungYokCheck == True:
    #     return ""

    test_result = Spacing.spacing(end_result)

    return test_result

def PreprocessingFile(fileName):
    start = datetime.now()
    totalCnt = 0

    print("Start PreprocessingFile " + fileName)
    print(start)

    f = open(fileName, 'r')
    print("Counting file")
    for line in f:
        totalCnt += 1
    print("Counting file success!")
    f.close()

    f = open(fileName, 'r')

    r = open("preprocessed_" + fileName, 'w+')
    cnt = 1

    
    while True:
        for line in f:
            # print(line)
            result = PreprocessString(line)
            # print(result)
            if len(result) <= 6:
                print("processing... too short pass {}/{}".format(cnt, totalCnt))
                cnt += 1
                continue

            r.writelines(result + '\n')
            print("processing... {}/{}".format(cnt, totalCnt))
            cnt += 1
        break

    f.close()
    r.close()
    end = datetime.now()
    print('{} work end, elapsed : {}, start : {}'.format(end, end - start, start))
    return

if __name__=='__main__':
    # PreprocessingFile("ilbe_reply_target.csv")
    result = PreprocessString("아이 시발 짜증나네 ㅋㅋㅋ")

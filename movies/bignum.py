# coding: utf-8
def sumup(left, right):
    print("left: ",end="")
    print(left)
    print("right: ",end="")
    print(right)
    buf = '0'
    res = []
    for l, r in zip_longest(left, right):
        if left and right:
            digits = [x for x in str(int(l) + int(r) + int(buf))]
            if len(digits) > 1:
                buf = '0'
                rev_digits = [x for x in reversed(digits)]
                print(rev_digits)
                lo = rev_digits[0]
                hi = rev_digits[1]
                res.append(lo)
                buf += hi
            else:
                res.append(digits)
        elif left == None:
            res.append(left)
        elif right == None:
            res.append(right)
            
    return res

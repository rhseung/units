# 2 1 3
# 2 1 2
# 1 1 2
# 1 1 1
# 0 1 1
# 0 0 1
# 0 0 0
a = [2, 1, 3, 7]

print(a)

while True:
    is_all_equal = True
    for i in range(1, len(a)):
        if a[i] != a[0]:
            is_all_equal = False
            break

    if is_all_equal and a[0] == 0:
        break

    if is_all_equal:
        a[0] -= 1
    else:
        max_i = 0
        for i in range(len(a)):
            if a[i] > a[max_i]:
                max_i = i

        a[max_i] -= 1

    print(a)

import argparse

"""
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
定义一个命令行参数。
参数name or flags是一个名称还是一个选项列表。
参数action是当遇到这个参数时怎么样处理。
        store 保存参数值，可能会先将参数值转换成另一个数据类型。若没有显式指定动作，则默认为该动作。
        store_const 保存一个被定义为参数规格一部分的值，而不是一个来自参数解析而来的值。这通常用于实现非布尔值的命令行标记。
        store_ture/store_false 保存相应的布尔值。这两个动作被用于实现布尔开关。
        append 将值保存到一个列表中。若参数重复出现，则保存多个值。
        append_const 将一个定义在参数规格中的值保存到一个列表中。
        version 打印关于程序的版本信息，然后退出
参数nargs是指这个命令要消耗多少个命令行参数。
参数const是有常量的需求，但不需要读取的。
参数default是当命令在命令行里不存在时默认的值。
参数type是命令行参数转换为那一种Python类型。
参数choices是这个命令参数允许的值的容器。
参数required是命令参数是否可选。
参数help是描述这个参数是做什么事情的。
参数metavar是在使用说明里参数的名称。
参数dest是参数在分析之后产生的对象里的属性名称。

ArgumentParser.parse_args(args=None, namespace=None)
转换参数字符串为命令行对象，并把相关参数转换为属性。
参数args是直接给出命令行参数，默认是使用系统的sys.argv参数；参数namespace是命名对象的名称。

"""
parser = argparse.ArgumentParser()

parser.add_argument('-s', action='store', dest='simple_value',
        help='Store a simple value')

parser.add_argument('-c', action='store_const', dest='constant_value',
        const='value-to-store',
        help='Store a constant value')

parser.add_argument('-t', action='store_true', default=False,
        dest='boolean_switch',
        help='Set a switch to true')
parser.add_argument('-f', action='store_false', default=False,
        dest='boolean_switch',
        help='Set a switch to false')

parser.add_argument('-a', action='append', dest='collection',
        default=[],
        help='Add repeated values to a list')

parser.add_argument('-A', action='append_const', dest='const_collection',
        const='value-1-to-append',
        default=[],
        help='Add different values to list')
parser.add_argument('-B', action='append_const', dest='const_collection',
        const='value-2-to-append',
        help='Add different values to list')

parser.add_argument('--version', action='version', version='%(prog)s 1.0')
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.") # 无默认值为None 表示参数可设置一个或多个
parser.add_argument('-a1', '--ign', nargs='?', type=str, help="ignore  list of classes.")    # 表示参数可设置零个或一个
parser.add_argument('-a2', '--ignu', nargs='*', type=str, help="ignore  list of classes.")   # 表示参数可设置一个或多个

results = parser.parse_args()
print('simple_value     =', results.simple_value)
print('constant_value   =', results.constant_value)
print('boolean_switch   =', results.boolean_switch)
print('collection       =', results.collection)
print('const_collection =', results.const_collection)
print('ignore =', results.ignore)
print('ign =', results.ign)
print('ignu =', results.ignu)

"""
# 只能空格链接不能用用别的符号
$ python test.py -i 1 2 3 4 5 
simple_value     = None
constant_value   = None
boolean_switch   = False
collection       = []
const_collection = []
ignore = ['1', '2', '3', '4', '5']

$  python test.py  -i 1 2 3 4 5 -a1 a
simple_value     = None
constant_value   = None
boolean_switch   = False
collection       = []
const_collection = []
ignore = ['1', '2', '3', '4', '5']
ignore = 'a'

$ python test.py  -i 1 2 3 4 5 -a1 a -a2 a b
simple_value     = None
constant_value   = None
boolean_switch   = False
collection       = []
const_collection = []
ignore = ['1', '2', '3', '4', '5']
ign = a
ignu = ['a', 'b']


$ python argparse_action.py -s value
simple_value     = value
constant_value   = None
boolean_switch   = False
collection       = []
const_collection = []

$ python argparse_action.py -c
simple_value     = None
constant_value   = value-to-store
boolean_switch   = False
collection       = []
const_collection = []

$ python argparse_action.py -t
simple_value     = None
constant_value   = None
boolean_switch   = True
collection       = []
const_collection = []

$ python argparse_action.py
simple_value     = None
constant_value   = None
boolean_switch   = False
collection       = []
const_collection = []

# 报错
$ python test.py -c => c是true(因为action)
$ python test.py    => c是false(default)
$ python argparse_action.py -t True
$ python argparse_action.py -t Flase


$ python argparse_action.py -f
simple_value     = None
constant_value   = None
boolean_switch   = False
collection       = []
const_collection = []

$ python argparse_action.py -a one -a two -a three
simple_value     = None
constant_value   = None
boolean_switch   = False
collection       = ['one', 'two', 'three']
const_collection = []

$ python argparse_action.py -B -A
simple_value     = None
constant_value   = None
boolean_switch   = False
collection       = []
const_collection = ['value-2-to-append', 'value-1-to-append']

"""

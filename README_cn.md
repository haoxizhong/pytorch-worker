# Pytorch Worker

本框架为基于pytorch的模型训练、测试框架。该框架的目的是方便大家快速上手写出pytorch的模型，同时能够定制化属于自己的模型、输出、数据处理和评价指标，方便大家快速完成同任务上的大量模型的实验。

英文的README可以通过[这里](https://github.com/haoxizhong/pytorch-worker/blob/master/README,md)访问。

## 目录

* [运行方法](#运行方法)
* [配置文件](#配置文件)
* [框架运行逻辑](!框架运行逻辑)
* [已有方法](#已有方法)
* [添加你自己的方法](#添加你自己的方法)
* [依赖库](#依赖库)
* [未来计划](#未来计划)
* [作者与致谢](#作者与致谢)

## 运行方法

运行方法分为模型训练和模型测试两个部分。

### 模型训练

无论是模型的训练还是测试，我们都需要指定本次运行的参数即配置文件，配置文件的详细说明可以可以参考[下一节](#配置文件)的内容。如果我们想要训练我们的模型，运行方法如下：

```bash
python3 train.py --config 配置文件 --gpu GPU列表
```

例如，如果我们想用编号为2,3,5的三张GPU运行中文Bert的分类任务，我们可以运行如下命令：

```bash
python3 train.py --config config/nlp/BasicBert.config --gpu 2,3,5
```

当然，如果你不想使用GPU来运行你的模型的话，你可以去掉``--gpu``选项来完成这一点，例如：

```bash
python3 train.py --config config/nlp/BasicBert.config
```

这样的运行方式就可以不使用GPU来训练模型。

如果你并不想从头开始训练你的模型，而是希望接着之前某次训练的结果继续运行的话，可以用如下命令：

```bash
python3 train.py --config 配置文件 --gpu GPU列表 --checkpoint 模型参数文件
```

参数``checkpoint``指向中间某次的训练结果文件，框架会从该文件中读取模型、优化器和训练迭代轮数等信息。（如果在中途修改了优化器并不会导致错误，框架会自动使用新的优化器继续运行）

### 模型测试

待补充。

## 配置文件

### 运行逻辑

配置文件是该框架的核心模块之一，绝大部分的运行参数都是通过读取配置文件得到的。我们以一个例子来说明框架从配置文件读取参数的运行逻辑：

```bash
python3 train.py --config config/nlp/BasicBert.config
```

在这段代码里面我们指定了从``BasicBert.config``中读取我们所需要的参数，在实际运行中我们会总共涉及到三个不同的配置文件：

1. ``config/nlp/BasicBert.config``
2. ``config/default_local.config``
3. ``config/default.config``

当框架尝试读取某个参数的时候，会按照上述三个配置文件的顺序从上至下依次读取。例如如果我们想要读取``batch_size``这个参数，框架会先尝试从``config/nlp/BasicBert.config``中读取该参数；如果失败，会再尝试从``config/default_local.config``读取该参数；如果再次失败，会尝试从``config/default.config``读取参数。如果在三个配置文件中都没有读取到参数，则会抛出异常。

对于三个配置文件，我们建议每个文件中所需要包含的参数有：

* ``config/default.config``：在这个文件中，我们建议将一些对于不同模型不变，或者说一些参数的默认值写在该文件中，例如像测试间隔、输出间隔等对于不同模型来说都不会有所改变的参数。
* ``config/default_local.config``：我们建议将模型所涉及到的路径信息写在该文件中，该文件并不会被同步到``git``中，更多情况下，该配置文件是用于在不同服务器上进行适配使用的文件。
* 运行指定的配置文件：我们建议将运行对应模型的相关参数写在该文件里，一些对于不同模型没有变化的参数如数据地址、数据处理方式等参数不建议写在该配置文件里面。

在程序运行中，我们传递的``config``参数便为所对应的配置文件，支持原版``ConfigParser``的各种函数包括但不限于``get,getint,getboolean``等方法。

### 基本参数说明

配置文件的结构是遵循``python``下的``ConfigParser``包进行构建的，文件中``[chapter]``代表的是不同适用情况的参数，具体结构可以参考``config``文件夹下的例子。我们接下来将会介绍在基本框架中所涉及到的一些参数的说明，参数分为必要参数（运行所有模型都需要的参数）和可选参数（运行特定模型所需要的参数）。当然，**你可以随意的在你自定义的方法里面增加新的参数**。

**[train]：训练用参数**

* ``epoch``：必要参数，代表需要训练的轮数。
* ``batch_size``：必要参数，代表训练时一次计算的数据量。
* ``shuffle``：必要参数，代表是否需要随机打乱数据。
* ``reader_num``：必要参数，需要多少个进程处理训练数据。
* ``optimizer``：必要参数，选择的优化器。
* ``learning_rate``：必要参数，学习率。
* ``weight_decay``：必要参数，权值正则化参数。
* ``step_size``和``lr_multiplier``：必要参数，学习率每过``step_size``个epoch变为原来的``lr_multiplier``倍。

**[eval]：测试用参数**

* ``batch_size``：必要参数，代表测试时一次计算的数据量。
* ``shuffle``：必要参数，代表是否需要随机打乱数据。
* ``reader_num``：必要参数，需要多少个进程处理测试数据。

**[data]：数据用参数**

* ``train_dataset_type,valid_dataset_type,test_dataset_type``：必要参数，分别代表训练、验证、测试时使用的[数据读取器](#数据读取器)类型。如果验证和测试的参数没有指定，则默认使用训练的类型。
* ``train_formatter_type,valid_formatter_type,test_formatter_type``：必要参数，分别代表训练、验证、测试时使用的[数据处理器](#数据处理器)类型。如果验证和测试的参数没有指定，则默认使用训练的类型。
* ``train_data_path,valid_data_path,test_data_path``：可选参数（仅用于框架已实现的数据读取器），分别代表训练、验证、测试时的数据位置。
* ``train_file_list,valid_file_list,test_file_list``：可选参数（仅用于框架已实现的数据读取器），分别代表训练、验证、测试对应的数据位置下，哪些文件或者文件夹属于数据。即真正的数据地址应该是``train_data_path+train_file_list``。
* ``recursive``：可选参数（仅用于``FilenameOnly,JsonFromFiles``两种数据读取器），代表如果对应的数据地址是一个文件夹，是否需要递归地向下搜索更多的数据。
* ``json_format``：可选参数（仅用于``ImageFromJson,JsonFromFiles``两种数据读取器），代表对应的``json``文件的格式。如果为``line``代表一行一个``json``数据；如果为``single``代表整个文件为一个``json``数据。
* ``load_into_mem``：可选参数（仅用于``ImageFromJson,JsonFromFiles``两种数据读取器），代表是否提前把所有数据加载到内存中。
* ``prefix``：可选参数（仅用于``ImageFromJson``数据读取器），代表图片的相对路径起始位置。

**[model]：模型用参数**

* ``model_name``：必要参数，代表训练的模型类型。
* ``bert_path``：可选参数（仅用于``BasicBert``模型），代表``bert``模型参数的地址。
* ``output_dim``：可选参数（仅用于框架已实现的模型），代表分类问题中模型所需要输出的种类数量。

**[output]：输出用参数**

* ``output_time``：必要参数，代表每多少次运行模型后输出一次结果。
* ``test_time``：必要参数，代表每多少个epoch进行一次验证。
* ``model_path``：必要参数，模型结果文件保存的地址。
* ``model_name``：必要参数，模型保存的名字。
* ``tensorboard_path``：必要参数，tensorboard存储的地址。（暂未实现）
* ``accuracy_method``：必要参数，计算模型好坏程度的[指标函数](#指标函数)。
* ``output_function``：必要参数，用来产生中间[指标输出](#指标输出)的函数。
* ``output_value``：可选参数（仅用于``Basic``版本的指标输出函数），用来选择要输出的指标。
* ``tqdm_ncols``：可选参数，代表框架运行时进度条的宽度，如果不选则默认为一行。

## 新方法的添加和已有方法

我们的框架中除开配置文件读取器以外，剩下的绝大部分模块都是可定制的，包括是：[数据读取器](#数据读取器)、[数据处理器](#数据处理器)、[模型](#模型)、[指标函数](#指标函数)、[指标输出](#指标输出)，这里每一个部分都可以添加你自己需要的方法或者模型，我们将在依次介绍每个模块的实现方法和功能。

### 数据读取器

**模块功能**：用于从文件中读取数据，存进pytorch的dataset中。

**实现方法**：如果需要实现新的数据读取器，我们需要在``dataset``文件加中新建一个文件来实现我们新的数据读取器，需要按照下列方法实现：

```python
from torch.utils.data import Dataset

class DatasetName(Dataset):
    def __init__(self, config, mode, *args, **params):
        # 在这里进行初始化
        # config为读取的配置文件
        # mode为读取器的模式，包括train、valid和test三种模式
        pass

    def __getitem__(self, item):
        # 返回第item个数据
        pass

    def __len__(self):
        # 返回数据的总量
        return len(self.file_list)
```

在实现好我们的数据读取器之后，再将实现的数据读取器添加到``dataset/__init__.py``的列表即可使用。你也可以通过已实现的方法来学习如何实现一个数据读取器。

**已实现的方法**：

* ``FilenameOnly``：只获取所有数据所对应的绝对路径的数据读取器。
* ``ImageFromJson``：通过一个``json``文件获取图片的地址和标签的数据读取器。``json``文件需要包含一个数组，数组里每个元素需要包括``path``（图片相对路径）和``label``（图片标签）两个字段。所谓相对路径，是以配置文件中``[data] prefix``所指定的路径为基础路径来说的。另外，该方法还可以通过改变``[data] load_into_mem``的值来决定是否提前将所有数据载入内存。
* ``JsonFromFiles``：从多个``json``文件中读取文本信息和标签的数据读取器。首先你可以通过设定``[data] json_format``来指定对应``json``文件的格式（见[基本参数说明](#基本参数说明)）。对于每条``json``数据，需要包含``text``（文本信息）和``label``（标签信息）。你同时也可以通过``[data] recursive``和``[data] load_into_mem``来决定是否递归查找文件和是否提前将数据加载进内存。注意，由于文本数据的特殊性，我们建议：**只有当提前将数据加载至内存时才对数据进行打乱操作，否则不要进行打乱操作，不然会大大降低数据读取的速度**。

### 数据处理器

**模块功能**：将数据读取器读取的数据处理成更够交给模型运行的格式。

**实现方法**：如果需要实现新的数据处理器，我们需要在``formatter``文件加中新建一个文件来实现我们新的数据处理器，需要按照下列方法实现：

```python
class FormatterName:
    def __init__(self, config, mode, *args, **params):
        # 在这里进行初始化
        # config为读取的配置文件
        # mode为处理器的模式，包括train、valid和test三种模式
        pass

    def process(self, data, config, mode, *args, **params):
        # 对给定的数据data进行处理
        # data的格式为大小为batch_size的数组（在test模式下，最后一个batch的大小可能小于batch_size），里面每个元素即为从数据处理器的__getitem__中返回的格式
        # config和mode参数的设置同上
        # 这里我们返回的数据类型要求必须为python的dict格式，且需要把模型需要的字段处理为Tensor的格式
        pass
```

在实现好我们的数据处理器之后，再将实现的数据处理器添加到``formatter/__init__.py``的列表即可使用。你也可以通过已实现的方法来学习如何实现一个数据处理器。

**已实现的方法**：

* ``Basic``：啥也不干的数据处理器，提供一个最基本的格式。
* ``BasicBert``：将``JsonFromFiles``读取的数据进行处理，把``text``转换为``BasicBert``模型所需要的token。

### 模型

**模块功能**：运行数据，产生结果。

**实现方法**：如果需要实现新的模型，我们需要在``model``文件加中新建一个文件来实现我们新的模型，需要按照下列方法实现：

```python
class ModelName(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        # 模型初始化部分
        # config为读取的配置文件
        # gpu_list为在运行的时候指定的gpu_id的列表
        super(ModelName, self).__init__()

    def init_multi_gpu(self, device, config, *args, **params):
        # 多卡初始化部分，用于将模型放置于多卡上
        # 如果没有多卡的需求，则不需要实现该函数
        # device为gpu_id的列表
        # config为读取的配置文件
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        # 模型运行的核心部分
        # data为数据处理器处理好的数据，已自动将其中的Tensor进行gpu化
        # config为读取的配置文件
        # gpu_list为在运行的时候指定的gpu_id的列表
        # acc_result为类型的评价指标结果，如已经运行的所有结果中的准确率、召回率等信息，由之后的指标函数所决定
        # mode为模型的模式，包括train、valid和test三种模式
        # 返回格式为要求为python的dict
        # 在train和valid模式中，由于需要衡量模型和优化模型，返回的结果中必须包含loss和acc_result两个字段，分别代表损失函数的结果和累计的指标量。acc_result的计算在这里并不是必须的，但是如果想从多维的角度评判模型请一定使用
        pass
```

在实现好我们的模型之后，再将实现的模型添加到``model/__init__.py``的列表即可使用。你也可以通过已实现的方法来学习如何实现一个模型。

**已实现的方法**：

- ``BasicBert``：基础的``Bert``单标签分类器。

### 指标函数

**模块功能**：产生除了损失以外的其他指标，用于衡量模型的水平。

**实现方法**：如果需要实现新的指标函数，我们需要在``tools/accuracy_tool.py``文件中新建一个方法来实现我们新的指标函数，需要按照下列方法实现：

```python
def FunctionName(outputs, label, config, result):
    # 这只是一个示例，实际上由于不同模型使用的评价指标都是不一样的，这里你可以随意改造参数，我们只以已经实现好的几个方法的参数进行说明
    # outputs为模型预测的结果
    # label为标签
    # config为读取的配置文件
    # result为历史累计的评价指标结果
    # 返回值为新的评价指标结果
    pass
```

在实现好我们的指标函数之后，再将实现的指标函数添加到``tools/accuracy_init.py``的列表即可使用。你也可以通过已实现的方法来学习如何实现一个指标函数。

**已实现的方法**：

- ``Null``：什么也不做的指标函数。
- ``SingleLabelTop1``：单标签分类问题的指标函数，用于计算每一类的``TP,TN,FP,FN``的值。
- ``MultiLabel``：多标签分类问题的指标函数，用于计算每一类的``TP,TN,FP,FN``的值。

### 指标输出

**模块功能**：通过指标函数产生的结果，产生用于打印至终端的评价指标。

**实现方法**：如果需要实现新的指标输出函数，我们需要在``tools/output_tool.py``文件中新建一个方法来实现我们新的指标输出函数，需要按照下列方法实现：

```python
def FunctionName(data, config, *args, **params):
    # data为我们使用指标函数产生的结果
    # config为读取的配置文件
    # 返回值为需要输出的指标结果，要求类型为字符串
```

在实现好我们的指标输出函数之后，再将实现的指标输出函数添加到``tools/output_init.py``的列表即可使用。你也可以通过已实现的方法来学习如何实现一个指标输出函数。

**已实现的方法**：

- ``Null``：什么也不做的指标输出函数。
- ``Basic``：分类方法的指标输出函数，可以选择输出``micro_precision,micro_recall,micro_f1,macro_precision,macro_recall,macro_f1``这六个指标中的任意多个。

## 框架运行逻辑

1. 读取配置文件。

2. 进行初始化操作。

   1）初始化数据处理器。

   2）初始化数据读取器。

   3）初始化模型，并多gpu化。

   4）初始化优化器。

   5）如果需要加载checkpoint，则进行加载。（加载出错只会显示warning）

3. 开始训练。

   训练分为训练和验证两个步骤，其中训练每次迭代的逻辑为：

   1）从数据读取器读取数据。

   2）将数据交给数据处理器进行处理。

   3）模型运行数据，产生损失和评价指标，并优化模型。

   4）如果需要输出评价指标，利用指标输出函数产生输出。

   5）返回第一步，完成一次迭代。

   一个epoch完成之后，保存模型，且会检查是否需要进行验证，验证流程与训练大致相同，不过只会在全部完成之后产生一次评价指标。

## 依赖库

请参考``requirements.txt``。（尚未完成）

## 未来计划

1. 添加对``Tensorboard``的支持。
2. 添加各种``loss``函数。
3. 添加对``lr_scheduler``的可定制化支持。
4. 优化进度条的显示模式。
5. 在各个可定制化模块中增加更多常用方法。

## 作者与致谢

暂无。
# ./common


```
common/
  |- callbacks/
  |- datasets/
  |- models/
  |     |- backbones/
  |- networks/
  |- __init__.py
  |- helpers.py
  |- logger.py
  |- README.md
```

- `callbacks` - Network különböző állapotaira (train/epohc elkezdődött, stb) hívható callbackek
- `datasets` - Adatforrásokkal, loaderekkel kapcsolatos általános osztályok
- `models` - Alapvető modellek (ResNet, Densenet, Egyszerű UNet), tesztekhez, kiindulási alapok
- `networks`- Újrahasznosítható hálózatok. A hálozatok fogják össze a train/inference pipeline-okat.
- `helpers.py` - Mindenféle
- `logger.py` - Local log (console, file, stb), neptune reporter

## Network
Minden feladat kiindulási alapja a `common.networks.AbstractNetwork`. Ez kezeli a teljes folyamatot. Főbb pontokban:
- Adatok, osztályok betöltése, kezelése (dataloaders, optimizers, schedulers, models, etc.)
- Kezeli a callback-eket
- Kezeli a train/validate/predict folyamatokat
- Kezeli az adatforrásokat (fit előtt be kell állítani a train/valid loader-eket)

`common.networks.classifiers.ImageClassifierNetwork` Spec hálózat képek osztályozásához.


### FIT (train)
`common.networks.AbstractNetwork#fit()`

Train folyamat
- A train hosszát nem epoch-ban számolja, hanem `iteration`-ben. 1 `iteration` megegyezik egy batch feldolgozásával
(forward). Így nagy mennyiségű adat esetén nem kell várni egy teljes epoch-ot pl validálásra.
- Megadott időközönként (iteration szám) futtat validálást (lásd az lenti validálással kapcsolatos callback-eket)
- Kezeli az APEX-et (nem teszteltem)
- Kezeli a megadott számú iteration utáni backprop-ot

## Callbacks
Minden hálózathoz tetszőleges számú `common.callbacks.Callback` adható hozzá. Ezek a hívások futnak:

##### on_train_begin(self, *args, **kwargs):
Train (fit) folyamat kezdetekor

##### on_train_end(self, *args, **kwargs):
Train (fit) folyamat befejezésekor

##### on_epoch_begin(self, epoch: int, *args, **kwargs):
Epoch kezdetekor (mielőtt az első `for ... TrainLoader` ciklus lefut)

##### on_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, *args, **kwargs):
Minden batch feldolgozás előtt.

##### on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor, targets: Tensor, loss: Tensor, *args, **kwargs):
Minden batch feldolgozás után. Ilyenkor már rendelkezésre áll a model output (általában logits) és a kapott loss érték
is (loss változtatás számít backprop-kor)

##### on_epoch_end(self, epoch: int, *args, **kwargs):
Epoch végén, miután az utolsó ```for ... train_loader``` batch is lefutott

##### on_validation_begin(self, epoch, iteration, *args, **kwargs):
Validálás kezdetekor, csak ha a megfelelő iteration fut (pl minden 500. iter esetén)

##### on_validation_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, *args, **kwargs):
Minden (valid) batch feldolgozás előtt

##### on_validation_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor, targets: Tensor, loss: Tensor, *args, **kwargs):
Minden (valid) batch feldolgozás után. Ilyenkor már rendelkezésre áll a model output (általában logits) és a kapott (valid)
loss érték is. (loss változtatás NEM számít backprop-kor )

##### on_validation_end(self, epoch, iteration, *args, **kwargs):
Validálás végén
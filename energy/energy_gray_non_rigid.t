-- --Dimension parameters
-- local W,H = Dim("W",0), Dim("H",1)	--Image size
-- local N = 	Dim("N",2)				--Number of vertices
-- 
-- --Problem parameters
-- local f_x = Param("f_x",float,0)
-- local f_y = Param("f_y",float,1)
-- local u_x = Param("u_x",float,2)
-- local u_y = Param("u_y",float,3)
-- 
-- local w_photometricSqrt = 	Param("w_photometricSqrt", float, 4)	--Photometric cost weight
-- local w_tvSqrt = 			Param("w_tvSqrt", float, 5)				--TV regularisation weight
-- local w_arapSqrt = 			Param("w_arapSqrt", float, 6)			--ARAP regularisation weight
-- local w_tempShapeSqrt = 	Param("w_tempShapeSqrt", float, 7)		--Temporal shape regularisation weight
-- 
-- local MeshTrans = 			Unknown("MeshTrans", float3,{N},8)			--vertex.xyz <- unknown
-- local Angle = 			Unknown("Angle", float3,{N},9)			--vertex.xyz, rotation.xyz <- unknown
-- 
-- local RigidRot_x = 		Param("RigidRot_x", float, 10)		--rigid rotation rotation.xyz <- unknown
-- local RigidRot_y = 		Param("RigidRot_y", float, 11)		--rigid rotation rotation.xyz <- unknown
-- local RigidRot_z = 		Param("RigidRot_z", float, 12)		--rigid rotation rotation.xyz <- unknown
-- local RigidRot = Vector(RigidRot_x, RigidRot_y, RigidRot_z )
-- 
-- local RigidTrans_x = 		Param("RigidTrans_x", float, 13)	--rigid trnaslation trnaslation.xyz <- unknown
-- local RigidTrans_y = 		Param("RigidTrans_y", float, 14)	--rigid trnaslation trnaslation.xyz <- unknown
-- local RigidTrans_z = 		Param("RigidTrans_z", float, 15)	--rigid trnaslation trnaslation.xyz <- unknown
-- local RigidTrans = Vector( RigidTrans_x, RigidTrans_y, RigidTrans_z )
-- 
-- local I_im = Image("I_im",float,{W,H},16) -- frame, sampled
-- local I_dx = Image("I_dx",float,{W,H},17) -- partials for frame
-- local I_dy = Image("I_dy",float,{W,H},18)
-- 
-- local TemplateColors = 	Array("TemplateColors", float, {N},19)	--template shape: vertex.xyz
-- local TemplateShape = 	Array("TemplateShape", float3, {N},20)	--template shape: vertex.xyz
-- local PrevMeshTrans =	Array("PrevMeshTrans", float3,{N},21)		--previous vertices MeshTrans: vertex.xyz
-- local Visibility = 		Array("Visibility", int, {N}, 22)			--Visibility
-- local G = Graph("G", 23, "v0", {N}, 24, "v1", {N}, 25)				--Graph
-- 
-- UsePreconditioner(true)
-- 
-- -- create a new math operator that samples from the image
-- local I = sampledimage(I_im, I_dx, I_dy)
-- 
-- function Intensity(v)
--     local x = v(0)
--     local y = v(1)
--     local z = v(2)
--     local i = (f_x * x - u_x * z) / z
--     local j = (f_y * y - u_y * z) / z
--     
--     return I(i,j)
-- end
-- 
-- function newVertex(v, dv, R, t)
-- 	return Rotate3D(R, v + dv) + t
-- end
-- 
-- --Photometric Error Data term
-- local new_vertex = newVertex(TemplateShape(G.v0), MeshTrans(G.v0), RigidRot, RigidTrans)
-- local photometricCost = Intensity(new_vertex) - TemplateColors(G.v0)
-- -- Energy( Select(Visibility(G.v0)>0, w_photometricSqrt*photometricCost, 0) )
-- Energy( w_photometricSqrt*photometricCost )
-- 
-- --TV regularization
-- local TVCost = MeshTrans(G.v0) - MeshTrans(G.v1)
-- Energy(w_tvSqrt*TVCost)
-- 
-- --ARAP regularization
-- local template_diff = TemplateShape(G.v0) - TemplateShape(G.v1)
-- local trans_diff = MeshTrans(G.v0) - MeshTrans(G.v1)
-- local ARAPCost = template_diff 
--                - Rotate3D( Angle(G.v0), template_diff + trans_diff )
-- Energy(w_arapSqrt*ARAPCost)
-- 
-- --Temporal Shape regularisation
-- local TempShapeCost = MeshTrans(G.v0) - PrevMeshTrans(G.v0)
-- Energy(w_tempShapeSqrt*TempShapeCost)

--Dimension parameters
local W,H = Dim("W",0), Dim("H",1)	--Image size
local N = 	Dim("N",2)				--Number of vertices

--Problem parameters
local w_photometricSqrt = 	Param("w_photometricSqrt", float, 0)	--Photometric cost weight
local MeshTrans = 			Unknown("MeshTrans", float3,{N},1)			--vertex.xyz <- unknown
local TemplateShape = 	Array("TemplateShape", float3, {N},2)	--template shape: vertex.xyz
local G = Graph("G", 3, "v0", {N}, 4, "v1", {N}, 5)				--Graph

-- UsePreconditioner(true)

--Photometric Error Data term
local photometricCost = TemplateShape(G.v0) - MeshTrans(G.v0)
Energy( w_photometricSqrt*photometricCost )
